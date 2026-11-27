import json

from jiwer import wer

from context_biasing.text_utils import fuzzy_fix, norm, present

d = json.load(open("preds_test.json"))
refs_raw, dicts = d["refs"], d["dicts"]
refs = [norm(t) for t in refs_raw]
total = sum(1 for r, ph in zip(refs_raw, dicts) for p in ph if present(r, p))


def report(label, preds):
    hits = sum(
        1
        for p, r, ph in zip(preds, refs_raw, dicts)
        for n in ph
        if present(r, n) and present(p, n)
    )
    print(
        f"RESULT {label:22s} recall: {hits}/{total}  WER: {wer(refs, [norm(p) for p in preds]):.4f}"
    )


base = d["baseline"]
fz = [fuzzy_fix(p, ph) for p, ph in zip(base, dicts)]
report("baseline", base)
report("baseline + fuzzy", fz)
report("NeMo", d["boosted_2.0"])
report("NeMo + fuzzy", d["fuzzy_2.0"])

good = bad = 0
bad_list = []
for i, (a, b) in enumerate(zip(base, fz)):
    for x, y in zip(a.split(" "), b.split(" ")):
        if x != y:
            ok = present(refs_raw[i], y)
            good += ok
            bad += not ok
            if not ok:
                bad_list.append((i, x, y))
print(
    f"RESULT swaps on baseline: {good + bad}, name in reference: {good}, not in reference: {bad}"
)
for i, x, y in bad_list[:20]:
    print(f"RESULT   clip {i}: {x} -> {y}")
