import json
import re

import jellyfish
from jiwer import wer

from context_biasing.text_utils import fuzzy_fix, lev, norm, present

d = json.load(open("preds_dev.json"))
refs_raw, dicts = d["refs"], d["dicts"]
base = d["boosted_2.0"]
refs = [norm(t) for t in refs_raw]
total = sum(1 for r, ph in zip(refs_raw, dicts) for p in ph if present(r, p))


def phon_fix(pred, phrases, max_ratio):
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
                if ck and ck == keys[n] and dist / max(len(n), len(core)) <= max_ratio:
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
        f"RESULT {label:28s} recall: {hits}/{total}  WER: {wer(refs, [norm(p) for p in preds]):.4f}"
    )


report("NeMo boosting a=2.0", base)
fz = [fuzzy_fix(p, ph) for p, ph in zip(base, dicts)]
report("+ edit-distance fuzzy", fz)
for ratio in [0.3, 0.45, 0.6]:
    out = [phon_fix(p, ph, ratio) for p, ph in zip(fz, dicts)]
    report(f"+ phonetic, ratio<={ratio}", out)
    if ratio == 0.45:
        for i, (a, b) in enumerate(zip(fz, out)):
            for x, y in zip(a.split(" "), b.split(" ")):
                if x != y:
                    print(
                        f"RESULT   clip {i}: {x} -> {y} | in reference: {present(refs_raw[i], y)}"
                    )
