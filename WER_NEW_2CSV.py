import pandas as pd
from jiwer import wer, Compose, RemoveMultipleSpaces, RemovePunctuation

# Define preprocessing for Sinhala
transform = Compose([
    RemoveMultipleSpaces(),
    RemovePunctuation()
])

# Load reference file
df_ref = pd.read_csv("reference.csv", encoding="utf-8", header=None, names=["filename", "reference"])
# Normalize filename to match
df_ref["filename"] = df_ref["filename"].apply(lambda x: x.split("/")[-1].strip())

# Load hypothesis file
df_hyp = pd.read_csv("hypothesis.csv", encoding="utf-8", header=None, names=["filename", "hypothesis"])

# Merge on filename
df = pd.merge(df_ref, df_hyp, on="filename", how="inner")

# Check
print(f"Loaded {len(df)} matched rows.")

# WER calculation
results = []
wers = []

for i, row in df.iterrows():
    ref_raw = str(row["reference"]).strip()
    hyp_raw = str(row["hypothesis"]).strip()

    # Clean text
    ref_clean = transform(ref_raw)
    hyp_clean = transform(hyp_raw)

    # Compute WER
    error = wer(ref_clean, hyp_clean)
    percent_error = error * 100
    wers.append(percent_error)

    results.append({
        "filename": row["filename"],
        "reference": ref_raw,
        "hypothesis": hyp_raw,
        "WER (%)": round(percent_error, 2)
    })

    print(f"{row['filename']}: WER = {percent_error:.2f}%")

# Save output CSV
df_out = pd.DataFrame(results)
df_out.to_csv("sinhala_asr_wer_output.csv", index=False, encoding="utf-8")

# Print average WER
average_wer = sum(wers) / len(wers)
print(f"\nAverage WER: {average_wer:.2f}%")
