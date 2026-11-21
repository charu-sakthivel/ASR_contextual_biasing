from datasets import load_dataset
import nemo.collections.asr as nemo_asr
from jiwer import wer
from openpyxl import Workbook

# Load dataset
dataset = load_dataset("argmaxinc/contextual-earnings22", split="test")

# Load Parakeet
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

# Create Excel workbook
workbook = Workbook()
sheet = workbook.active
sheet.title = "Baseline Results"

# Column headers
sheet.append(
    ["File ID", "Ground Truth", "Parakeet Prediction", "Dictionary", "Keywords", "WER"]
)

# Process every sample
for i, sample in enumerate(dataset):
    print(f"Processing {i + 1}/{len(dataset)}")

    # Get audio
    audio = sample["audio"].get_all_samples()
    audio_tensor = audio.data.squeeze()

    # Run Parakeet
    output = model.transcribe([audio_tensor])

    prediction = output[0].text

    # Calculate WER
    error_rate = wer(sample["text"], prediction)

    # Add result to Excel
    sheet.append(
        [
            sample["file_id"],
            sample["text"],
            prediction,
            ", ".join(sample["dictionary"]),
            ", ".join(sample["keywords"]),
            error_rate,
        ]
    )

# Save Excel file
workbook.save("baseline_results.xlsx")

print("\nDone!")
print("Saved as baseline_results.xlsx")
