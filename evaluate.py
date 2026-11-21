import argparse
import copy
import json
import re

import nemo.collections.asr as nemo_asr
from datasets import load_dataset
from jiwer import wer
from omegaconf import open_dict

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="dev")
parser.add_argument("--alphas", default="1.0,1.5,2.0,2.5")
parser.add_argument("--beam", type=int, default=4)
args = parser.parse_args()
ALPHAS = [float(a) for a in args.alphas.split(",")]


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def present(text, ph):
    return " " + norm(ph) + " " in " " + norm(text) + " "


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fuzzy_fix(pred, phrases):
    names = [p for p in phrases if " " not in p]
    out = []
    for w in pred.split(" "):
        pre, core, post = re.match(r"^(\W*)(.*?)((?:['’]s)?\W*)$", w).groups()
        if core and core[0].isupper():
            best = None
            for n in names:
                if core.lower() == n.lower():
                    best = None
                    break
                d = lev(core.lower(), n.lower())
                limit = 1 if len(n) <= 5 else 2
                if d <= limit and (best is None or d < best[0]):
                    best = (d, n)
            if best:
                core = best[1]
        out.append(pre + core + post)
    return " ".join(out)


def best_text(out):
    first = out[0]
    best = first[0] if isinstance(first, list) else first
    return best.text


dataset = load_dataset("argmaxinc/contextual-earnings22", split=args.split)
N = len(dataset)
print(f"split={args.split} clips={N}")

model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)
orig_cfg = copy.deepcopy(model.cfg.decoding)


def audio_of(i):
    return dataset[i]["audio"].get_all_samples().data.squeeze()


refs_raw = [dataset[i]["text"] for i in range(N)]
dicts = [list(dataset[i]["dictionary"]) for i in range(N)]
refs = [norm(t) for t in refs_raw]
total = sum(1 for r, d in zip(refs_raw, dicts) for ph in d if present(r, ph))


def real_hits(preds):
    return sum(
        1
        for p, r, d in zip(preds, refs_raw, dicts)
        for ph in d
        if present(r, ph) and present(p, ph)
    )


def report(label, preds):
    print(
        f"RESULT {label:24s} recall: {real_hits(preds)}/{total}  WER: {wer(refs, [norm(p) for p in preds]):.4f}"
    )


base_preds = []
for i in range(N):
    print(f"baseline {i + 1}/{N}")
    base_preds.append(best_text(model.transcribe([audio_of(i)])))
report("baseline", base_preds)

saved = {
    "split": args.split,
    "beam": args.beam,
    "refs": refs_raw,
    "dicts": dicts,
    "baseline": base_preds,
}

for alpha in ALPHAS:
    boosted = []
    for i in range(N):
        print(f"alpha {alpha}: {i + 1}/{N}")
        if not dicts[i]:
            boosted.append(base_preds[i])
            continue
        cfg = copy.deepcopy(orig_cfg)
        with open_dict(cfg):
            cfg.strategy = "malsd_batch"
            cfg.beam.beam_size = args.beam
            cfg.beam.allow_cuda_graphs = False
            cfg.beam.boosting_tree = {
                "key_phrases_list": dicts[i],
                "context_score": 1.0,
                "depth_scaling": 2.0,
            }
            cfg.beam.boosting_tree_alpha = alpha
        model.change_decoding_strategy(cfg)
        boosted.append(best_text(model.transcribe([audio_of(i)])))
    fixed = [fuzzy_fix(p, d) for p, d in zip(boosted, dicts)]
    report(f"boosted a={alpha}", boosted)
    report(f"boosted+fuzzy a={alpha}", fixed)
    saved[f"boosted_{alpha}"] = boosted
    saved[f"fuzzy_{alpha}"] = fixed

with open(f"preds_{args.split}.json", "w") as f:
    json.dump(saved, f)
print(f"RESULT saved preds_{args.split}.json")
