from melo.api import TTS
from transformers import pipeline
import numpy as np
import gradio as gr

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
def transcribe(stream, new_chunk):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]


def main():
    print('ready to start app')
    demo = gr.Interface(
        transcribe,
        ["state", gr.Audio(sources=["microphone"], streaming=True)],
        ["state", "text"],
        live=True,
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