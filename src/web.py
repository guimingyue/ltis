
from flask import Flask, request, jsonify
#from chat import Chat
from pydub import AudioSegment
import numpy as np
from datetime import datetime


app = Flask(__name__)
#chat = Chat('chinese', 'qwen_turbo', '你是一个知识渊博的助手', 'base', 'ZH')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.post('/api/media')
def media():
    file = request.files['audio']
    audio = AudioSegment.from_file_using_temporary_files(file.stream, format='wave')
    data = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        data = data.reshape(-1, audio.channels)
    t = audio.frame_rate, data
    file_path = f'{datetime.now().strftime("%Y%m%d%H%M%S")}.wav'
    file.save(file_path)
    
    '''speech, text = chat.transcribe(t)
    return {
        "speech": speech,
        "text": text,
        "file_type": file.content_type,
    }'''
    return jsonify({
        'text': 'xxxxx',
        'url': file_path
    })