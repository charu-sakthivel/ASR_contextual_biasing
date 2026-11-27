import copy
import json
import random

import jiwer
import nemo.collections.asr as nemo_asr
from datasets import load_dataset
from omegaconf import open_dict

from context_biasing.text_utils import fuzzy_fix, norm, present

USE_DROPOUT, NUM = True, 5  # fixed in advance, do not change
random.seed(0)

saved = json.load(open("preds_test.json"))
refs_raw, dicts = saved["refs"], saved["dicts"]
N = len(refs_raw)

dataset = load_dataset("argmaxinc/contextual-earnings22", split="test")
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)
orig_cfg = copy.deepcopy(model.cfg.decoding)


def best_text(out):
    first = out[0]
    best = first[0] if isinstance(first, list) else first
    return best.text


preds = []
for i in range(N):
    print(f"variants: {i + 1}/{N}")
    if not dicts[i]:
        preds.append(saved["baseline"][i])
        continue
    cfg = copy.deepcopy(orig_cfg)
    with open_dict(cfg):
        cfg.strategy = "malsd_batch"
        cfg.beam.beam_size = 4
        cfg.beam.allow_cuda_graphs = False
        cfg.beam.boosting_tree = {
            "key_phrases_list": dicts[i],
            "context_score": 1.0,
            "depth_scaling": 2.0,
            "use_bpe_dropout": USE_DROPOUT,
            "num_of_transcriptions": NUM,
            "bpe_alpha": 0.3,
        }
        cfg.beam.boosting_tree_alpha = 2.0
    model.change_decoding_strategy(cfg)
    audio = dataset[i]["audio"].get_all_samples().data.squeeze()
    preds.append(best_text(model.transcribe([audio])))

fixed = [fuzzy_fix(p, d) for p, d in zip(preds, dicts)]
json.dump(
    {"variants": preds, "variants_fuzzy": fixed}, open("preds_bpe_test.json", "w")
)

systems = {
    "NeMo": saved["boosted_2.0"],
    "variants": preds,
    "NeMo+fuzzy": saved["fuzzy_2.0"],
    "variants+fuzzy": fixed,
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
n = len(stats["NeMo"])


def metrics(idx, k):
    rows = [stats[k][j] for j in idx]
    return (
        sum(r[0] for r in rows) / sum(r[1] for r in rows),
        sum(r[2] for r in rows) / sum(r[3] for r in rows),
    )


def ci(v):
    v = sorted(v)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


pairs = [("variants", "NeMo"), ("variants+fuzzy", "NeMo+fuzzy")]
diffs = {p: ([], []) for p in pairs}
for _ in range(2000):
    idx = [random.randrange(n) for _ in range(n)]
    res = {k: metrics(idx, k) for k in systems}
    for x, y in pairs:
        diffs[(x, y)][0].append(res[x][0] - res[y][0])
        diffs[(x, y)][1].append(res[x][1] - res[y][1])

full = list(range(n))
for k in systems:
    w, r = metrics(full, k)
    print(f"RESULT {k:15s} recall {r:.4f}  WER {w:.4f}")
for x, y in pairs:
    wl, wh = ci(diffs[(x, y)][0])
    rl, rh = ci(diffs[(x, y)][1])
    print(
        f"RESULT {x} minus {y}: recall diff [{rl:+.4f}, {rh:+.4f}]   WER diff [{wl:+.4f}, {wh:+.4f}]"
    )
