import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

texts = [
    "Anna",
    "Call Anna tomorrow",
    "Prayanova",
    "Call Prayanova tomorrow",
    "Ericsson",
    "Call Ericsson tomorrow",
]

for text in texts:
    token_ids = model.tokenizer.text_to_ids(text)
    print(f"{text} -> {token_ids}")
