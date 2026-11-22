import json
import random

import jiwer

from context_biasing.text_utils import norm, present

random.seed(0)

a = json.load(open("preds_test.json"))
b = json.load(open("preds_own_test.json"))
refs_raw, dicts = a["refs"], a["dicts"]
systems = {
    "baseline": a["baseline"],
    "NeMo": a["boosted_2.0"],
    "own": b["own"],
    "NeMo+fuzzy": a["fuzzy_2.0"],
    "own+fuzzy": b["own_fuzzy"],
}

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
    return (
        sum(r[0] for r in rows) / sum(r[1] for r in rows),
        sum(r[2] for r in rows) / sum(r[3] for r in rows),
    )


def ci(vals):
    vals = sorted(vals)
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


pairs = [("own", "NeMo"), ("own+fuzzy", "NeMo+fuzzy")]
B = 2000
boot = {k: {"wer": [], "rec": []} for k in systems}
diffs = {p: {"wer": [], "rec": []} for p in pairs}
for _ in range(B):
    idx = [random.randrange(n) for _ in range(n)]
    res = {k: metrics(idx, k) for k in systems}
    for k, (w, r) in res.items():
        boot[k]["wer"].append(w)
        boot[k]["rec"].append(r)
    for x, y in pairs:
        diffs[(x, y)]["wer"].append(res[x][0] - res[y][0])
        diffs[(x, y)]["rec"].append(res[x][1] - res[y][1])

full = list(range(n))
for k in systems:
    w, r = metrics(full, k)
    wl, wh = ci(boot[k]["wer"])
    rl, rh = ci(boot[k]["rec"])
    print(
        f"RESULT {k:11s} recall {r:.3f} [{rl:.3f}, {rh:.3f}]   WER {w:.4f} [{wl:.4f}, {wh:.4f}]"
    )

for x, y in pairs:
    rl, rh = ci(diffs[(x, y)]["rec"])
    wl, wh = ci(diffs[(x, y)]["wer"])
    print(
        f"RESULT {x} minus {y}: recall diff [{rl:+.4f}, {rh:+.4f}]   WER diff [{wl:+.4f}, {wh:+.4f}]"
    )
