from pydub import AudioSegment
import os
from pathlib import Path

# Set the path to your WAV file
input_file = Path(r"C:\Users\Sajini Sandeepa\OneDrive\Desktop\Semester 07\EE74099  EN74099 Research Project\Self Recoding Sinhala Audio Datasets\news and lec type recodings\Extracted audios(not edit)\05_25_slbcnews_new.wav")  # Update this path if needed

# Output folder for chunks
output_folder = input_file.parent / "chunks_1"
output_folder.mkdir(exist_ok=True)

# Load the WAV file
audio = AudioSegment.from_wav(input_file)

# Split duration settings (in milliseconds)
min_len = 10 * 1000  # 15 sec
max_len = 20 * 1000  # 30 sec

# Loop to split into chunks
chunk_count = 0
start = 0
while start < len(audio):
    end = min(start + max_len, len(audio))
    chunk = audio[start:end]
    if len(chunk) >= min_len:
        chunk_filename = output_folder / f"{input_file.stem}_part{chunk_count + 1}.wav"
        chunk.export(chunk_filename, format="wav")
        print(f"✔️ Saved: {chunk_filename.name}")
        chunk_count += 1
    start += max_len

print(f"\n✅ Done! Total {chunk_count} chunks saved to: {output_folder}")
