import json
import random
import re

import jiwer

random.seed(0)


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def present(text, ph):
    return " " + norm(ph) + " " in " " + norm(text) + " "


d = json.load(open("preds_test.json"))
refs_raw, dicts = d["refs"], d["dicts"]
systems = {
    "baseline": d["baseline"],
    "boosted": d["boosted_2.0"],
    "boosted+fuzzy": d["fuzzy_2.0"],
}

# per-clip statistics: (word errors, ref words, name hits, names in ref)
stats = {k: [] for k in systems}
for i, ref_raw in enumerate(refs_raw):
    ref = norm(ref_raw)
    if not ref:
        continue
    nref = len(ref.split())
    tot = sum(1 for ph in dicts[i] if present(ref_raw, ph))
    for k, preds in systems.items():
        hyp = norm(preds[i])
        if hyp:
            o = jiwer.process_words(ref, hyp)
            errs = o.substitutions + o.deletions + o.insertions
        else:
            errs = nref
        hits = sum(
            1 for ph in dicts[i] if present(ref_raw, ph) and present(preds[i], ph)
        )
        stats[k].append((errs, nref, hits, tot))

n = len(stats["baseline"])


def metrics(idx, k):
    rows = [stats[k][j] for j in idx]
    e = sum(r[0] for r in rows)
    w = sum(r[1] for r in rows)
    h = sum(r[2] for r in rows)
    t = sum(r[3] for r in rows)
    return e / w, h / t


def ci(vals):
    vals = sorted(vals)
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


B = 2000
boot = {k: {"wer": [], "rec": []} for k in systems}
d_wer, d_rec = [], []
for _ in range(B):
    idx = [random.randrange(n) for _ in range(n)]
    res = {k: metrics(idx, k) for k in systems}
    for k, (w, r) in res.items():
        boot[k]["wer"].append(w)
        boot[k]["rec"].append(r)
    d_wer.append(res["boosted"][0] - res["boosted+fuzzy"][0])
    d_rec.append(res["boosted+fuzzy"][1] - res["boosted"][1])

full = list(range(n))
for k in systems:
    w, r = metrics(full, k)
    wl, wh = ci(boot[k]["wer"])
    rl, rh = ci(boot[k]["rec"])
    print(
        f"RESULT {k:14s} recall {r:.3f} [{rl:.3f}, {rh:.3f}]   WER {w:.4f} [{wl:.4f}, {wh:.4f}]"
    )

lo, hi = ci(d_rec)
print(f"RESULT fuzzy minus boosted, recall diff: [{lo:+.4f}, {hi:+.4f}]")
lo, hi = ci(d_wer)
print(
    f"RESULT boosted minus fuzzy, WER diff (positive = fuzzy better): [{lo:+.4f}, {hi:+.4f}]"
)
