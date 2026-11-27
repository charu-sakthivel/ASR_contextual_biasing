import json
import re

import jellyfish
from jiwer import wer

from context_biasing.text_utils import fuzzy_fix, lev, norm, present

RATIO = 0.45  # fixed in advance, do not change

d = json.load(open("preds_test.json"))
refs_raw, dicts = d["refs"], d["dicts"]
refs = [norm(t) for t in refs_raw]
total = sum(1 for r, ph in zip(refs_raw, dicts) for p in ph if present(r, p))


def phon_fix(pred, phrases):
    names = [p for p in phrases if " " not in p]
    keys = {n: jellyfish.metaphone(n) for n in names}
    out = []
    for w in pred.split(" "):
        pre, core, post = re.match(r"^(\W*)(.*?)((?:['’]s)?\W*)$", w).groups()
        if core and core[0].isupper() and len(core) >= 4:
            ck = jellyfish.metaphone(core)
            for n in names:
                if core.lower() == n.lower():
                    break
                dist = lev(core.lower(), n.lower())
                if ck and ck == keys[n] and dist / max(len(n), len(core)) <= RATIO:
                    core = n
                    break
        out.append(pre + core + post)
    return " ".join(out)


def report(label, preds):
    hits = sum(
        1
        for p, r, ph in zip(preds, refs_raw, dicts)
        for n in ph
        if present(r, n) and present(p, n)
    )
    print(
        f"RESULT {label:24s} recall: {hits}/{total}  WER: {wer(refs, [norm(p) for p in preds]):.4f}"
    )


fz = d["fuzzy_2.0"]
ph_out = [phon_fix(p, ph) for p, ph in zip(fz, dicts)]
report("NeMo + fuzzy", fz)
report("NeMo + fuzzy + phonetic", ph_out)

good = bad = 0
for i, (a, b) in enumerate(zip(fz, ph_out)):
    for x, y in zip(a.split(" "), b.split(" ")):
        if x != y:
            ok = present(refs_raw[i], y)
            good += ok
            bad += not ok
            print(f"RESULT   clip {i}: {x} -> {y} | in reference: {ok}")
print(f"RESULT swaps in reference: {good}, not in reference: {bad}")
