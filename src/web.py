import soundfile
from flask import Flask, request, jsonify, send_file, Response, stream_with_context, after_this_request, render_template
from chat import Chat
from pydub import AudioSegment
import numpy as np
from datetime import datetime
from flask_cors import CORS
import time
import os


app = Flask(__name__)
CORS(app)
chat = Chat('english')
#chat = Chat('chinese', 'qwen-turbo', '你是一个知识渊博的助手', 'base', 'ZH')

base_audio_file_dir = os.getcwd() + '/audio_files'
os.makedirs(base_audio_file_dir, exist_ok=True)


@app.route("/")
def hello_world():
    return render_template("index.html")

@app.post('/api/media')
def media():
    file = request.files['audio']
    audio = AudioSegment.from_file_using_temporary_files(file.stream, format='wave')
    data = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        data = data.reshape(-1, audio.channels)

    src_file = f'src_{datetime.now().strftime("%Y%m%d%H%M%S")}.wav'
    soundfile.write(f'{base_audio_file_dir}/{src_file}', data, audio.frame_rate)
    res_file = f'res_{datetime.now().strftime("%Y%m%d%H%M%S")}.wav'

    '''
    speech = data
    soundfile.write(f'{base_audio_file_dir}/{res_file}', speech, audio.frame_rate)
    return jsonify({
        "src_text": 'xxxxxxx',
        "src_audio": src_file,
        "speech": res_file,
        "text": 'yyyyyy',
        "timestamp": int(time.time() * 1000)
    })
    '''

    src_text, text = chat.transcribe((audio.frame_rate, data), f'{base_audio_file_dir}/{res_file}')
    
    return jsonify({
        "src_text": src_text,
        "src_audio": src_file,
        "speech": res_file,
        "text": text,
        "timestamp": int(time.time() * 1000)
    })
   
@app.route('/audio/<filename>')
def stream_audio(filename):
    # 确保文件路径正确，并且指向音频文件存储的位置

    file_path = f'{base_audio_file_dir}/{filename}'
    @after_this_request
    def stream_audio(response):
        # 设置正确的MIME类型
        response.headers['Content-Type'] = 'audio/mpeg'
        # 其他可能需要的HTTP头
        response.headers['Accept-Ranges'] = 'bytes'
        response.status_code = 200
        return response
    return send_file(file_path)
