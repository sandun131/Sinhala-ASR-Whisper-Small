import pandas as pd
from jiwer import wer

# Load reference CSV without header, assign columns
df_ref = pd.read_csv("Transcription_openslr5(evalutaion).csv", 
                     header=None, names=["filename", "reference"])

# Remove the folder prefix to match second CSV filenames
df_ref["filename"] = df_ref["filename"].apply(lambda x: x.split('/')[-1])  # keep only the file name, e.g., 0dd6a06488.wav

# Load hypothesis CSV without header and assign columns
df_hyp = pd.read_csv("Transcription_openslr5(evl)_using our model.csv",
                     header=None, names=["filename", "hypothesis"])

# Merge on the cleaned 'filename' column
df = pd.merge(df_ref, df_hyp, on="filename")

# Calculate WER for each pair using jiwer library (more robust)
df["wer"] = df.apply(lambda row: wer(str(row["reference"]), str(row["hypothesis"])), axis=1)

# Calculate overall average WER
overall_wer = df["wer"].mean()
#print(f"Overall Word Error Rate (WER): {overall_wer:.4f}")
print(f"Overall Word Error Rate (WER): {overall_wer * 100:.2f}%")
