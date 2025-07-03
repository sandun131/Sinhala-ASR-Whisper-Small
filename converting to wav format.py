import os
import subprocess
from pathlib import Path

#Set the directory where your .webm files are
input_folder = Path(r"C:\Users\Sajini Sandeepa\OneDrive\Desktop\Semester 07\EE74099  EN74099 Research Project\Self Recoding Sinhala Audio Datasets\news and lec type recodings\webm format files\drama_1.webm")  # Change to your actual folder
output_folder =Path(r'C:\Users\Sajini Sandeepa\OneDrive\Desktop\Semester 07\EE74099  EN74099 Research Project\Self Recoding Sinhala Audio Datasets\news and lec type recodings\wav format files')  # Or use Path("C:/output")

# Loop through all .webm files in the folder
for webm_file in input_folder.glob("*.webm"):
    wav_file = output_folder / (webm_file.stem + ".wav")

    # FFmpeg command
    command = [
        "ffmpeg",
        "-i", str(webm_file),
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        str(wav_file)
    ]

    print(f"Converting: {webm_file.name} ➜ {wav_file.name}")
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

print("✅ Conversion complete!")
