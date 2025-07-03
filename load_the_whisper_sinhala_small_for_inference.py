from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load model and processor
model = WhisperForConditionalGeneration.from_pretrained("./whisper-sinhala-small")
processor = WhisperProcessor.from_pretrained("./whisper-sinhala-small")

# Load the audio file as raw waveform
audio_path = r"E:\FYP_19_11\data set\fyp11\datasets\collected dataset 4\evaluation\25_02_17_slbcnews_clean_part64.wav"
waveform, sample_rate = torchaudio.load(audio_path)

# Resample if needed (Whisper expects 16000 Hz)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Convert stereo to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0).unsqueeze(0)

# Prepare inputs
inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

# Generate transcription
generated_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Transcription:", transcription)
