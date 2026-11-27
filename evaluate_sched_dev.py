import copy
import json

import nemo.collections.asr as nemo_asr
import nemo.collections.asr.parts.submodules.rnnt_decoding as rd
from datasets import load_dataset
from jiwer import wer
from omegaconf import open_dict

from context_biasing.text_utils import fuzzy_fix, norm, present
from context_biasing.trie_fusion import TrieFusionModel

GRID = [
    ("nemo-shape c0=1 a=2.0", 1.0, 2.0),
    ("nemo-shape c0=1 a=1.0", 1.0, 1.0),
]  # (label, c0, alpha)

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


report("NeMo a=2.0 (saved)", saved["boosted_2.0"])
report(
    "own power b=1 g=0.5 a=2.0 (earlier)",
    json.load(open("preds_own_dev2.json"))["own_2.0_0.5"],
)

dataset = load_dataset("argmaxinc/contextual-earnings22", split="dev")
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)
orig_cfg = copy.deepcopy(model.cfg.decoding)
audios = [dataset[i]["audio"].get_all_samples().data.squeeze() for i in range(N)]


class TrieShim:
    c0 = 1.0

    @classmethod
    def from_config(cls, cfg, tokenizer=None):
        tok = tokenizer if tokenizer is not None else model.tokenizer
        phrases = getattr(cfg, "key_phrases_list", None) or cfg["key_phrases_list"]
        return TrieFusionModel(
            list(phrases), tok, tok.vocab_size, bonus=cls.c0, schedule="nemo"
        )


def best_text(out):
    first = out[0]
    best = first[0] if isinstance(first, list) else first
    return best.text


rd.GPUBoostingTreeModel = TrieShim
for label, c0, alpha in GRID:
    TrieShim.c0 = c0
    preds = []
    for i in range(N):
        print(f"{label}: {i + 1}/{N}")
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
            }
            cfg.beam.boosting_tree_alpha = alpha
        model.change_decoding_strategy(cfg)
        preds.append(best_text(model.transcribe([audios[i]])))
    report(label, preds)
    report(label + " +fuzzy", [fuzzy_fix(p, d) for p, d in zip(preds, dicts)])
