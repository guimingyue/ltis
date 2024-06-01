from tts import MeloTts
from sr import WhisperSr
from llm import Qwen

lang_accent_sr_tts = {
    # language accent: (whisper language, melotts language, language)
    "EN-US": ("en", "EN", "english"),
    "EN-BR": ("en", "EN", "english"),
    "ZH": ("zh", "ZH", "chinese"),
    "JP": ("ja", "JP", "japanese"),
}

class Chat:
    def __init__(self,
                 name,
                 llm_model_name="qwen-max",
                 llm_prompt="You are a helpful assistant.",
                 sr_model_name="base",
                 accent="EN-US"):
        self.name = name
        lang = lang_accent_sr_tts[accent]
        self.sr = WhisperSr(sr_model_name, lang[0])
        self.tts = MeloTts(lang[1], accent)
        self.llm = Qwen(llm_model_name, llm_prompt)

    def transcribe(self, audio, path=None):
        src_text = self.sr.transcribe(audio)
        print("src text is: " + src_text)
        res_text, status = self.llm.call_with_messages(src_text)
        print("text from llm: " + res_text)
        if not status:
            return 
        
        self.tts.tts(res_text, path)
        return (src_text, res_text)