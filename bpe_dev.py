import copy
import json

import nemo.collections.asr as nemo_asr
from datasets import load_dataset
from jiwer import wer
from omegaconf import open_dict

from context_biasing.text_utils import fuzzy_fix, norm, present

saved = json.load(open("preds_dev.json"))
refs_raw, dicts = saved["refs"], saved["dicts"]
N = len(refs_raw)
refs = [norm(t) for t in refs_raw]
total = sum(1 for r, d in zip(refs_raw, dicts) for ph in d if present(r, ph))


def report(label, preds):
    hits = sum(
        1
        for p, r, d in zip(preds, refs_raw, dicts)
        for ph in d
        if present(r, ph) and present(p, ph)
    )
    print(
        f"RESULT {label:30s} recall: {hits}/{total}  WER: {wer(refs, [norm(p) for p in preds]):.4f}"
    )


def best_text(out):
    first = out[0]
    best = first[0] if isinstance(first, list) else first
    return best.text


dataset = load_dataset("argmaxinc/contextual-earnings22", split="dev")
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)
orig_cfg = copy.deepcopy(model.cfg.decoding)
audios = [dataset[i]["audio"].get_all_samples().data.squeeze() for i in range(N)]

report("NeMo a=2.0 (saved)", saved["boosted_2.0"])

SETTINGS = [(True, 5), (True, 5), (True, 5)]  # three repeats of the same setting
for use_dropout, num in SETTINGS:
    preds = []
    for i in range(N):
        print(f"bpe_dropout={use_dropout} num={num}: {i + 1}/{N}")
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
                "use_bpe_dropout": use_dropout,
                "num_of_transcriptions": num,
                "bpe_alpha": 0.3,
            }
            cfg.beam.boosting_tree_alpha = 2.0
        model.change_decoding_strategy(cfg)
        preds.append(best_text(model.transcribe([audios[i]])))
    fixed = [fuzzy_fix(p, d) for p, d in zip(preds, dicts)]
    report(f"NeMo+dropout n={num}", preds)
    report(f"NeMo+dropout n={num} +fuzzy", fixed)
