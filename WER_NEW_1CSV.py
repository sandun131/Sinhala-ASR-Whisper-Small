import pandas as pd
from jiwer import wer, Compose, RemoveMultipleSpaces, RemovePunctuation

# Create transformation for Sinhala normalization
transform = Compose([
    RemoveMultipleSpaces(),   # Remove double/multiple spaces
    RemovePunctuation()       # Remove punctuation (including Sinhala/English)
])

# Load CSV with reference & hypothesis
df = pd.read_csv("input.csv", encoding="utf-8")  # Make sure your file is UTF-8 encoded

# List to store WERs
wers = []

# Loop through each row
for index, row in df.iterrows():
    reference = str(row['reference']).strip()
    hypothesis = str(row['hypothesis']).strip()

    # Apply normalization (optional)
    ref_clean = transform(reference)
    hyp_clean = transform(hypothesis)

    # Compute WER
    error = wer(ref_clean, hyp_clean)
    percent_error = error * 100
    wers.append(percent_error)

    print(f"Row {index + 1}: WER = {percent_error:.2f}%")

# Add WER column to DataFrame
df["WER (%)"] = [round(w, 2) for w in wers]

# Save updated CSV
df.to_csv("output_with_wer.csv", index=False, encoding="utf-8")

# Print average WER
average_wer = sum(wers) / len(wers)
print(f"\nAverage WER: {average_wer:.2f}%")

