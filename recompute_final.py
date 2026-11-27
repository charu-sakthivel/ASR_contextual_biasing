import json
import random

import jiwer

from context_biasing.text_utils import fuzzy_fix, norm, present

random.seed(0)

a = json.load(open("preds_test.json"))
b = json.load(open("preds_own_test.json"))
c = json.load(open("preds_own_test2.json"))
refs_raw, dicts = a["refs"], a["dicts"]


def fz(preds):
    return [fuzzy_fix(p, d) for p, d in zip(preds, dicts)]


systems = {
    "baseline": a["baseline"],
    "baseline+fuzzy": fz(a["baseline"]),
    "NeMo": a["boosted_2.0"],
    "NeMo+fuzzy": fz(a["boosted_2.0"]),
    "own": b["own"],
    "own+fuzzy": fz(b["own"]),
    "own a2": c["own"],
    "own a2+fuzzy": fz(c["own"]),
}

stats = {k: [] for k in systems}
for i, ref_raw in enumerate(refs_raw):
    ref = norm(ref_raw)
    if not ref:
        continue
    nref = len(ref.split())
    tot = sum(1 for ph in dicts[i] if present(ref_raw, ph))
    for k, p in systems.items():
        hyp = norm(p[i])
        if hyp:
            o = jiwer.process_words(ref, hyp)
            errs = o.substitutions + o.deletions + o.insertions
        else:
            errs = nref
        hits = sum(1 for ph in dicts[i] if present(ref_raw, ph) and present(p[i], ph))
        stats[k].append((errs, nref, hits, tot))
n = len(stats["baseline"])


def metrics(idx, k):
    rows = [stats[k][j] for j in idx]
    return (
        sum(r[0] for r in rows) / sum(r[1] for r in rows),
        sum(r[2] for r in rows) / sum(r[3] for r in rows),
    )


def ci(v):
    v = sorted(v)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


pairs = [
    ("baseline+fuzzy", "baseline"),
    ("NeMo", "baseline"),
    ("NeMo+fuzzy", "NeMo"),
    ("own", "NeMo"),
    ("own+fuzzy", "NeMo+fuzzy"),
]
boot = {k: ([], []) for k in systems}
diffs = {p: ([], []) for p in pairs}
for _ in range(2000):
    idx = [random.randrange(n) for _ in range(n)]
    res = {k: metrics(idx, k) for k in systems}
    for k, (w, r) in res.items():
        boot[k][0].append(w)
        boot[k][1].append(r)
    for x, y in pairs:
        diffs[(x, y)][0].append(res[x][0] - res[y][0])
        diffs[(x, y)][1].append(res[x][1] - res[y][1])

full = list(range(n))
for k in systems:
    w, r = metrics(full, k)
    wl, wh = ci(boot[k][0])
    rl, rh = ci(boot[k][1])
    print(
        f"RESULT {k:15s} recall {r:.3f} [{rl:.3f}, {rh:.3f}]  WER {w:.4f} [{wl:.4f}, {wh:.4f}]"
    )
for x, y in pairs:
    wl, wh = ci(diffs[(x, y)][0])
    rl, rh = ci(diffs[(x, y)][1])
    print(
        f"RESULT {x} minus {y}: recall [{rl:+.4f}, {rh:+.4f}]  WER [{wl:+.4f}, {wh:+.4f}]"
    )
