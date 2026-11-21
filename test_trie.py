import nemo.collections.asr as nemo_asr

from context_biasing.trie import TrieBiasing

model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)
tok = model.tokenizer

names = ["Pryanova", "Gazprom", "Anna"]
bias = TrieBiasing(names, tok, bonus=2.0)

for name in names:
    ids = tok.text_to_ids(name)
    state = bias.init_state()
    total = 0.0
    for tid in ids:
        delta, state = bias.step(state, tid)
        total += delta
    print(f"{name}: ids={ids} total_bonus={total}")

# Partial match then a wrong token: the bonus must be revoked
ids = tok.text_to_ids("Pryanova")
state = bias.init_state()
total = 0.0
for tid in [ids[0], 999]:
    delta, state = bias.step(state, tid)
    total += delta
print(f"partial match then wrong token: total_bonus={total}")
