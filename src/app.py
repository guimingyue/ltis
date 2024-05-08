
import gradio as gr
from llm import Qwen
from sr import WhisperSr
from tts import MeloTts
import collections
from chat import Chat

dict = collections.defaultdict()
dict['user_en'] = Chat('english')
dict['user_zh'] = Chat('chinese', 'qwen_turbo', '你是一个知识渊博的助手', 'base', 'ZH')


def transcribe(audio, user='user_zh'):
    chat = dict[user]
    return chat.transcribe(audio)[0] 

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