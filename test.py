from src.stt import Whisper

def main():
    whisper_instance = Whisper()
    with whisper_instance:
        # Assuming you have an audio file named 'sample_audio.mp3' in the current directory
        audio_file_path = 'female.wav'
        transcription = whisper_instance.transcribe(audio_file_path)
        print("Transcription:", transcription)

if __name__ == "__main__":
    main()