
import gradio as gr
from llm import Qwen
from sr import WhisperSr
from tts import MeloTts

llm = Qwen('qwen_turbo', prompt='You are an American assistant who speak American English.')
sr = WhisperSr()
tts = MeloTts()


def transcribe(audio):
    text = sr.transcribe(audio)
    print("src text is: " + text)
    text, status = llm.call_with_messages(text)
    print("text from llm: " + text)
    if not status:
        return 
    
    return tts.tts(text) 

def main():
    print('starting app')
    demo = gr.Interface(
        transcribe,
        gr.Audio(sources=["microphone"]),
        "audio",
    )
    demo.launch()

if __name__ == "__main__":
    main()