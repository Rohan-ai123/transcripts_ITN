from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch
import os

# Function to split audio into chunks
def split_audio(audio_path, chunk_duration_ms=60000):
    """
    Splits audio file into smaller chunks of specified duration.
    Args:
        audio_path (str): Path to the audio file.
        chunk_duration_ms (int): Duration of each chunk in milliseconds (default 60 seconds).
    Returns:
        List of audio chunks.
    """
    audio = AudioSegment.from_file(audio_path)
    chunks = [audio[i:i+chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]
    return chunks

# Function to process and transcribe audio chunks
def transcribe_audio_chunks(chunks, model, processor, sampling_rate=16000):
    """
    Transcribes each audio chunk.
    Args:
        chunks (list): List of audio chunks.
        model: Whisper model instance.
        processor: Whisper processor instance.
        sampling_rate (int): Sampling rate for preprocessing.
    Returns:
        List of transcriptions.
    """
    transcriptions = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        # Export chunk to a temporary WAV file
        chunk_path = f"chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        # Load audio with librosa
        audio, rate = librosa.load(chunk_path, sr=sampling_rate)
        inputs = processor(audio, sampling_rate=rate, return_tensors="pt")

        # Transcribe the audio chunk
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        output = model.generate(inputs["input_features"], forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(output, skip_special_tokens=True)[0]

        transcriptions.append(transcription)

        # Clean up temporary file
        os.remove(chunk_path)

    return transcriptions

# Main function
def main(audio_path):
    # Load model and processor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

    # Split the audio into chunks
    print("Splitting audio into chunks...")
    chunks = split_audio(audio_path, chunk_duration_ms=60000)  # Adjust chunk duration as needed
    print(f"Total chunks created: {len(chunks)}")

  
    print("Starting transcription...")
    transcriptions = transcribe_audio_chunks(chunks, model, processor)


    full_transcription = " ".join(transcriptions)
    print("Transcription completed.")
    return full_transcription


audio_path = "/content/20240228_161902_AAGL-leilani.andorferlopez%40nextiva.com-1287-IN.wav"

transcription = main(audio_path)
print("Full Transcription:")
print(transcription)
