from melo.api import TTS
from transformers import pipeline
import numpy as np
import gradio as gr
import soundfile as sf
import io

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    text = transcriber({"sampling_rate": sr, "raw": y})["text"]
    print("text is: " + text)
    language='EN'
    accent='EN-US'

    # CPU is sufficient for real-time inference.
    # You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
    device = 'auto' # Will automatically use GPU if available
    melo_model = TTS(language=language, device=device)
    speaker_ids = melo_model.hps.data.spk2id
    audio = melo_model.tts_to_file(text, speaker_ids[accent], None, speed=1.0)
    print("convert success," + text)
    audio_buf = io.BytesIO()
    audio_buf.name = 'file.wav'
    sf.write(audio_buf, audio, melo_model.hps.data.sampling_rate)
    audio_buf.seek(0)  # Necessary for `.read()` to return all bytes
    return audio_buf.read()


def main():
    print('ready to start app')
    demo = gr.Interface(
        transcribe,
        gr.Audio(sources=["microphone"]),
        "audio",
    )
    demo.launch()
    print('ready to start')
    """
    # Speed is adjustable
    speed = 1.0

    language='EN'
    accent='EN-US'
    output_path='f.wav'
    text="What's your problem"

    # CPU is sufficient for real-time inference.
    # You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
    device = 'auto' # Will automatically use GPU if available
    
    melo_model = TTS(language=language, device=device)
    speaker_ids = melo_model.hps.data.spk2id
    audio = melo_model.tts_to_file(text, speaker_ids[accent], None, speed=speed)
    
    print("trans voice to text")
    whisper_model = whisper.load_model("base")
    text = whisper_model.transcribe(audio)
    
    print(text["text"])
    """


if __name__ == "__main__":
    main()