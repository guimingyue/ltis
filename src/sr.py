from transformers import pipeline
import numpy as np

# whisper models, see the release of whisper in hauggingface https://huggingface.co/openai
whisper_transcriber_base_en = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
whisper_transcriber_base = pipeline("automatic-speech-recognition", model="openai/whisper-base")

class WhisperSr:
    def __init__(self, model_name="base", language="en", verbose=False) -> None:
        if model_name == "base":
            if language == "en":
                self.model = whisper_transcriber_base_en
            else:
                self.model = whisper_transcriber_base
        else:
            self.model = whisper_transcriber_base
        self.verbose = verbose

    def transcribe(self, audio) -> str:
        sr, y = audio
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        trans_res = self.model({"sampling_rate": sr, "raw": y})
        return trans_res["text"]
        # result = transcriber(audio, language=self.language, verbose=self.verbose)
