import difflib
import json
from collections import defaultdict

import nemo.collections.asr as nemo_asr

from context_biasing.text_utils import norm, present

a = json.load(open("preds_test.json"))
b = json.load(open("preds_own_test.json"))
refs, dicts = a["refs"], a["dicts"]
nemo, own = a["boosted_2.0"], b["own"]

model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)
tok = model.tokenizer


def pieces(name):
    return tok.ids_to_tokens(tok.text_to_ids(name))


def closest(name, pred):
    n = norm(name)
    ws = norm(pred).split()
    return max(
        ws, key=lambda w: difflib.SequenceMatcher(None, n, w).ratio(), default=""
    )


by_len = defaultdict(lambda: [0, 0, 0])
only_own = []
for i, d in enumerate(dicts):
    for ph in d:
        if not present(refs[i], ph):
            continue
        k = min(len(pieces(ph)), 6)
        by_len[k][0] += 1
        o, n = present(own[i], ph), present(nemo[i], ph)
        by_len[k][1] += o
        by_len[k][2] += n
        if n and not o:
            only_own.append((ph, pieces(ph), closest(ph, own[i])))

print("RESULT pieces | names | my recall | NeMo recall")
for k in sorted(by_len):
    t, o, n = by_len[k]
    lab = f"{k}+" if k == 6 else str(k)
    print(f"RESULT {lab:>6} | {t:5d} | {o / t:.3f} | {n / t:.3f}")

print("RESULT --- names only my trie misses (NeMo gets them) ---")
for ph, pc, w in only_own[:30]:
    print(f"RESULT {ph} {pc} -> mine wrote: {w}")
