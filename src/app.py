from melo.api import TTS
from transformers import pipeline
import numpy as np
import gradio as gr
import soundfile as sf
import io
import dashscope
from http import HTTPStatus

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
messages = []

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    text = transcriber({"sampling_rate": sr, "raw": y})["text"]
    print("src text is: " + text)

    text, status = call_with_messages(text)
    if not status:
        return 

    print("text from llm:" + text)

    language='EN'
    accent='EN-US'
    # CPU is sufficient for real-time inference.
    # You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
    device = 'auto' # Will automatically use GPU if available
    melo_model = TTS(language=language, device=device)
    speaker_ids = melo_model.hps.data.spk2id
    audio = melo_model.tts_to_file(text, speaker_ids[accent], None, speed=1.0)
    audio_buf = io.BytesIO()
    audio_buf.name = 'file.wav'
    sf.write(audio_buf, audio, melo_model.hps.data.sampling_rate)
    audio_buf.seek(0)  # Necessary for `.read()` to return all bytes
    return audio_buf.read()

'''
you need to save api key
'''
def call_with_messages(text):
    messages.append({'role': 'user', 'content': text})

    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        result = response.output.choices[0]
        messages.append({'role': result['message']['role'],
                         'content': result['message']['content']})
        return result['message']['content'], True
    else:
        messages[:-1]
        return 'EOF ERROR', False
def init(prompt):
    messages.append({'role': 'system', 'content': prompt})

def main():
    print('starting')
    init('You are an American assistant who speak American English.')
    demo = gr.Interface(
        transcribe,
        gr.Audio(sources=["microphone"]),
        "audio",
    )
    demo.launch()

if __name__ == "__main__":
    main()