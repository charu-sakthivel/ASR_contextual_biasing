import json

import nemo.collections.asr as nemo_asr

from context_biasing.trie_fusion import TrieFusionModel


class FakeTok:
    """Tokens are letters: 'a b c' -> [0, 1, 2]."""

    def text_to_ids(self, s):
        return [ord(ch) - ord("a") for ch in s.replace(" ", "").lower()]


# Phrases 'abc' and 'bcd': after reading 'a','b','c' then 'd', fallback
# should land on the 'bcd' match; the no-fallback trie resets to the root.
ph = ["abc", "bcd"]
off = TrieFusionModel(ph, FakeTok(), 26, bonus=1.0, fallback=False)
on = TrieFusionModel(ph, FakeTok(), 26, bonus=1.0, fallback=True)
diff = int(
    (off.next_table != on.next_table).sum() + (off.score_table != on.score_table).sum()
)
print(f"RESULT synthetic example: {diff} differing entries (must be > 0)")

# Real dev dictionaries
tok = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
).tokenizer
dicts = json.load(open("preds_dev.json"))["dicts"]
changed = 0
for d in dicts:
    if not d:
        continue
    a = TrieFusionModel(d, tok, tok.vocab_size, bonus=2.0, gamma=0.5, fallback=False)
    b = TrieFusionModel(d, tok, tok.vocab_size, bonus=2.0, gamma=0.5, fallback=True)
    if (a.next_table != b.next_table).any() or (a.score_table != b.score_table).any():
        changed += 1
print(
    f"RESULT dev clips whose tables differ with fallback on: {changed} of {len(dicts)}"
)
