import difflib
import json
from collections import Counter

from context_biasing.text_utils import norm, present

a = json.load(open("preds_test.json"))
b = json.load(open("preds_own_test.json"))
refs, dicts = a["refs"], a["dicts"]
nemo, own = a["boosted_2.0"], b["own"]


def classify(name, pred):
    n = norm(name)
    if " " in n:
        return "multi-word name", ""
    words = norm(pred).split()
    best = max(
        words, key=lambda w: difflib.SequenceMatcher(None, n, w).ratio(), default=""
    )
    ratio = difflib.SequenceMatcher(None, n, best).ratio() if best else 0.0
    if ratio >= 0.75:
        return "near-miss (spelling variant)", best
    tail = n[-4:]
    if len(n) >= 6 and any(tail in w and not w.startswith(n[:2]) for w in words):
        return "start wrong, end right", best
    if any(w.startswith(n[:3]) for w in words):
        return "start right, end wrong", best
    return "no trace of the name", best


cats = Counter()
examples = {}
missed_names = Counter()
only_own_miss = only_nemo_miss = both_miss = 0
for i, d in enumerate(dicts):
    for ph in d:
        if not present(refs[i], ph):
            continue
        own_hit, nemo_hit = present(own[i], ph), present(nemo[i], ph)
        if not own_hit and nemo_hit:
            only_own_miss += 1
        if own_hit and not nemo_hit:
            only_nemo_miss += 1
        if not own_hit and not nemo_hit:
            both_miss += 1
        if not own_hit:
            cat, best = classify(ph, own[i])
            cats[cat] += 1
            missed_names[ph] += 1
            examples.setdefault(cat, []).append((ph, best))

total_miss = sum(cats.values())
print(f"RESULT names missed by my trie: {total_miss}")
for c, k in cats.most_common():
    print(f"RESULT   {c}: {k} ({100 * k / total_miss:.0f}%)  e.g. {examples[c][:4]}")
print(
    f"RESULT missed by mine only: {only_own_miss}, by NeMo only: {only_nemo_miss}, by both: {both_miss}"
)
print("RESULT most-missed names:", missed_names.most_common(12))
