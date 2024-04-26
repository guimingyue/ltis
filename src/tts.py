from melo.api import TTS
import soundfile as sf
import io

class MeloTts:
    def __init__(self, language='EN', accent='EN-US', speed=1.0) -> None:
        # CPU is sufficient for real-time inference.
        # You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
        device = 'auto' # Will automatically use GPU if available
        self.melo_model = TTS(language=language, device=device)
        self.speaker_ids = self.melo_model.hps.data.spk2id
        self.speed = speed
        self.accent = accent
    
    def tts(self, text):
        audio = self.melo_model.tts_to_file(text, self.speaker_ids[self.accent], None, self.speed)
        audio_buf = io.BytesIO()
        audio_buf.name = 'file.wav'
        sf.write(audio_buf, audio, self.melo_model.hps.data.sampling_rate)
        audio_buf.seek(0)  # Necessary for `.read()` to return all bytes
        return audio_buf.read()