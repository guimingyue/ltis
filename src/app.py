from melo.api import TTS
import whisper

def main():
    print('ready to start')
    # Speed is adjustable
    speed = 1.0

    language='EN'
    accent='EN-US'
    output_path='f.wav'
    text="What's your problem"

    # CPU is sufficient for real-time inference.
    # You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
    device = 'auto' # Will automatically use GPU if available

    melo_model = TTS(language=language, device=device)
    speaker_ids = melo_model.hps.data.spk2id
    melo_model.tts_to_file(text, speaker_ids[accent], output_path, speed=speed)
    
    print("trans voice to text")
    whisper_model = whisper.load_model("base")
    text = whisper_model.transcribe(output_path)
    print(text["text"])



if __name__ == "__main__":
    main()