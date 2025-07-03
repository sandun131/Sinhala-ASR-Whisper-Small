# finetune_whisper_sinhala.py

import os
import time
import torch
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

LANGUAGE = "Sinhala"
TASK = "transcribe"
MODEL_NAME = "openai/whisper-small"
SAMPLE_RATE = 16000
CSV_PATH = r"E:\FYP_19_11\python files\Transcription_openslr2.csv"
MAX_INPUT_LENGTH = 448
MAX_LABEL_LENGTH = 448  # Whisper's max decoder length

# Step 1: Load CSV dataset
def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path, names=["audio", "transcription"])  # Add `header=None` if CSV has no headers
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return dataset

# Step 2: Load processor & model
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

# Step 3: Preprocessing function
def preprocess(batch):
    audio = batch["audio"]

    # Extract audio features
    input_features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]

    # Tokenize transcription and mask padding for loss
    labels = processor.tokenizer(
        batch["transcription"],
        padding="max_length",
        max_length=MAX_LABEL_LENGTH,
        truncation=True,
        return_tensors="pt"
    ).input_ids[0]
    
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Important for ignoring loss on padding

    return {
        "input_features": input_features,
        "labels": labels
    }

# Step 4: Load and preprocess dataset
dataset = load_csv_dataset(CSV_PATH)
dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Optional: Split the dataset
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Step 5: Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-sinhala-small",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="no",
    max_steps=1000,
)

# Step 6: Initialize trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,  # Fixed: use processor, not feature_extractor
)
# --- Step 7: Train and save with time measurement ---
start_time = time.time()
trainer.train()
end_time = time.time()

# Calculate and display elapsed time
elapsed = end_time - start_time
minutes, seconds = divmod(elapsed, 60)
print(f"\n✅ Fine-tuning completed in {int(minutes)} minutes and {int(seconds)} seconds.\n")

# Step 8: Train and save
model.save_pretrained("whisper-sinhala-small")
processor.save_pretrained("whisper-sinhala-small")
