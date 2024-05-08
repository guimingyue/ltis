import json
from flask import Flask, request
from chat import Chat


app = Flask(__name__)
chat = Chat('chinese', 'qwen_turbo', '你是一个知识渊博的助手', 'base', 'ZH')

@app.route("/api")
def hello_world():
    return "<p>Hello, World!</p>"

@app.post('/api/media')
def media():
    file = request.files['audio']
    speech, text = chat.transcribe(file)
    return {
        "speech": speech,
        "text": text
    }

