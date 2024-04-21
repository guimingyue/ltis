from transformers import pipeline
import numpy as np

transcriber_en = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

class WhisperSr:
    def __init__(self, model="tiny", language="zh", verbose=False) -> None:
        self.model = model
        self.language = language
        self.verbose = verbose

    def transcribe(self, audio) -> str:
        sr, y = audio
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        trans_res = transcriber_en({"sampling_rate": sr, "raw": y})
        return trans_res["text"]
        # result = transcriber(audio, language=self.language, verbose=self.verbose)
