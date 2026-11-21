import copy
import re

import nemo.collections.asr as nemo_asr
from datasets import load_dataset
from jiwer import wer
from omegaconf import open_dict

N = 30
START = 30
BEAM = 4


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def best_text(out):
    first = out[0]
    best = first[0] if isinstance(first, list) else first
    return best.text


def hits(pred, phrases):
    p = " " + norm(pred) + " "
    return sum(1 for ph in phrases if " " + norm(ph) + " " in p)


dataset = load_dataset("argmaxinc/contextual-earnings22", split="test")
samples = [dataset[i] for i in range(START, START + N)]
audios = [s["audio"].get_all_samples().data.squeeze() for s in samples]

model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)
orig_cfg = copy.deepcopy(model.cfg.decoding)

# Baseline
base_preds = []
for i, a in enumerate(audios):
    print(f"baseline {i + 1}/{N}")
    base_preds.append(best_text(model.transcribe([a])))

refs = [norm(s["text"]) for s in samples]
total = sum(len(s["dictionary"]) for s in samples)
base_hits = sum(hits(p, s["dictionary"]) for p, s in zip(base_preds, samples))
print(
    f"\nRESULT baseline recall: {base_hits}/{total}  WER: {wer(refs, [norm(p) for p in base_preds]):.4f}"
)


def run_boosted(alpha):
    preds = []
    for i, (a, s) in enumerate(zip(audios, samples)):
        phrases = list(s["dictionary"])
        if not phrases:
            preds.append(base_preds[i])
            continue
        cfg = copy.deepcopy(orig_cfg)
        with open_dict(cfg):
            cfg.strategy = "malsd_batch"
            cfg.beam.beam_size = BEAM
            cfg.beam.allow_cuda_graphs = False
            cfg.beam.boosting_tree = {
                "key_phrases_list": phrases,
                "context_score": 1.0,
                "depth_scaling": 2.0,
            }
            cfg.beam.boosting_tree_alpha = alpha
        model.change_decoding_strategy(cfg)
        preds.append(best_text(model.transcribe([a])))
    return preds


for alpha in [1.5]:
    print(f"running alpha {alpha}")
    preds = run_boosted(alpha)
    h = sum(hits(p, s["dictionary"]) for p, s in zip(preds, samples))
    print(
        f"RESULT alpha={alpha}  recall: {h}/{total}  WER: {wer(refs, [norm(p) for p in preds]):.4f}"
    )

print("\n--- MISSED NAMES (alpha 1.5) ---")
preds = run_boosted(1.5)
for i, (p, s) in enumerate(zip(preds, samples)):
    missed = [
        ph for ph in s["dictionary"] if " " + norm(ph) + " " not in " " + norm(p) + " "
    ]
    if missed:
        print(f"RESULT clip {START + i} missed {missed}")
        print(f"RESULT   REF:  {s['text']}")
        print(f"RESULT   PRED: {p}")
