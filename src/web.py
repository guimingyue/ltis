import soundfile
from flask import Flask, request, jsonify
from chat import Chat
from pydub import AudioSegment
import numpy as np
from datetime import datetime
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
chat = Chat('chinese', 'qwen_turbo', '你是一个知识渊博的助手', 'base', 'ZH')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.post('/api/media')
def media():
    file = request.files['audio']
    audio = AudioSegment.from_file_using_temporary_files(file.stream, format='wave')
    #audio = AudioSegment.from_file_using_temporary_files(file.stream)
    data = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        data = data.reshape(-1, audio.channels)

    file_path = f'{datetime.now().strftime("%Y%m%d%H%M%S")}.wav'
    '''
    speech = data
    soundfile.write(file_path, speech, audio.frame_rate)
    return jsonify({
        'text': 'xxxxx',
        'url': file_path
    })
    '''
    speech, text = chat.transcribe(audio.frame_rate, data)
    soundfile.write(file_path, speech, audio.frame_rate)
    return {
        "speech": speech,
        "text": text,
        "file_type": file.content_type,
    }
    