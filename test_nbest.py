import re

import nemo.collections.asr as nemo_asr
from datasets import load_dataset
from omegaconf import open_dict

CLIPS = {36: "Bayon", 51: "Trip.com", 55: "ISA", 46: "Hepsipapel"}
BEAM = 16
TOP = 8


def norm(t):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split())


dataset = load_dataset("argmaxinc/contextual-earnings22", split="test")
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

cfg = model.cfg.decoding
with open_dict(cfg):
    cfg.strategy = "malsd_batch"
    cfg.beam.beam_size = BEAM
    cfg.beam.allow_cuda_graphs = False
    cfg.beam.return_best_hypothesis = False
model.change_decoding_strategy(cfg)

for idx, target in CLIPS.items():
    audio = dataset[idx]["audio"].get_all_samples().data.squeeze()
    out = model.transcribe([audio])[0]
    hyps = out.n_best_hypotheses if hasattr(out, "n_best_hypotheses") else out
    print(f"RESULT clip {idx} target={target} n_best={len(hyps)}")
    found = False
    for rank, h in enumerate(hyps):
        has = " " + norm(target) + " " in " " + norm(h.text) + " "
        found = found or has
        if rank < TOP:
            print(
                f"RESULT   #{rank} score={float(h.score):.2f} has_target={has}: {h.text[-90:]}"
            )
    print(f"RESULT   target anywhere in n-best: {found}")
