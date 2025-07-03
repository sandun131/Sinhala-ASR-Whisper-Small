import os
from pydub import AudioSegment, effects, silence
import noisereduce as nr
import numpy as np
from scipy.io import wavfile
from pydub.silence import split_on_silence
from pathlib import Path

# ----------- SETTINGS -----------
input_folder = r"C:\Users\Sajini Sandeepa\OneDrive\Desktop\Semester 07\EE74099  EN74099 Research Project\Self Recoding Sinhala Audio Datasets\news and lec type recodings\Extracted audios(not edit)\chunks_1"
output_folder = r"C:\Users\Sajini Sandeepa\OneDrive\Desktop\Semester 07\EE74099  EN74099 Research Project\Self Recoding Sinhala Audio Datasets\news and lec type recodings\Radio News\chunks files of 05_25_slbcnews"

# ----------- PROCESSING LOOP -----------
for file in Path(input_folder).glob("*.wav"):
    print(f"🔄 Processing {file.name}")

    # Load and preprocess audio
    sound = AudioSegment.from_wav(file)
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound = effects.normalize(sound)
    chunks = silence.split_on_silence(sound, min_silence_len=500, silence_thresh=-40.0)
    trimmed = sum(chunks)  # Optional: combine all non-silent chunks back to one AudioSegment

    # Save temp trimmed file
    temp_path = Path(output_folder) / f"temp_{file.stem}.wav"
    os.makedirs(temp_path.parent, exist_ok=True)
    trimmed.export(temp_path, format="wav")

    # Apply noise reduction
    rate, data = wavfile.read(temp_path)
    reduced = nr.reduce_noise(y=data, sr=rate)

    # Save final output
    final_path = Path(output_folder) / f"{file.stem}_clean.wav"
    wavfile.write(final_path, rate, reduced.astype(np.int16))

    # Delete temp file
    temp_path.unlink()

    print(f"✅ Saved: {final_path.name}")

print("🎉 All files processed successfully.")

