import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Patch deprecated numpy types
np.complex = complex
np.float = float

# Folder containing audio files (change this to your folder path)
audio_folder = r"D:\fyp11\datasets\Selected final clips"
output_folder = r"D:\fyp11\datasets\selected final clips mfcc and spectrograms"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each WAV file in the folder
for filename in os.listdir(audio_folder):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(audio_folder, filename)
        print(f"Processing: {file_path}")

        # Load audio
        y, sr = librosa.load(file_path, sr=None)

        # Extract MFCCs and Spectrogram
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        D = np.abs(librosa.stft(y))
        DB = librosa.amplitude_to_db(D, ref=np.max)

        # Create figure
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # Plot MFCC
        img1 = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax[0])
        ax[0].set(title='MFCC')
        fig.colorbar(img1, ax=ax[0], format="%+2.f dB")

        # Plot Spectrogram
        img2 = librosa.display.specshow(DB, x_axis='time', y_axis='log', sr=sr, ax=ax[1])
        ax[1].set(title='Spectrogram')
        fig.colorbar(img2, ax=ax[1], format="%+2.f dB")

        # Save figure
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_folder, f"{base_name}_features.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")
