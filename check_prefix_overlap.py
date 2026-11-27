import json

import nemo.collections.asr as nemo_asr

from context_biasing.text_utils import present

tok = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
).tokenizer
d = json.load(open("preds_test.json"))
refs_raw, dicts = d["refs"], d["dicts"]

clips = 0
spoken_pairs = []
for i, names in enumerate(dicts):
    seqs = {}
    for p in names:
        for v in {p, p.lower(), p.title()}:
            seqs[tuple(tok.text_to_ids(v))] = p
    hit = set()
    for a, pa in seqs.items():
        for b, pb in seqs.items():
            if pa != pb and len(a) < len(b) and b[: len(a)] == a:
                hit.add((pa, pb))
    if hit:
        clips += 1
    for short, long_ in hit:
        if present(refs_raw[i], long_):
            spoken_pairs.append((i, short, long_))

print(
    f"RESULT clips with a phrase that is a prefix of another: {clips} of {len(dicts)}"
)
print(
    f"RESULT spoken names that are the LONGER phrase in such a pair: {len(spoken_pairs)} (of 1300 spoken names)"
)
for i, s, l in spoken_pairs[:15]:
    print(f"RESULT   clip {i}: '{s}' is a prefix of '{l}'")
