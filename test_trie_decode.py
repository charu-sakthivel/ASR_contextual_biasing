import copy

import nemo.collections.asr as nemo_asr
import nemo.collections.asr.parts.submodules.rnnt_decoding as rd
from datasets import load_dataset
from omegaconf import open_dict

from context_biasing.trie_fusion import TrieFusionModel

CLIPS = [0, 36, 51]
BONUSES = [0.5, 1.0, 1.5, 2.0, 3.0]

dataset = load_dataset("argmaxinc/contextual-earnings22", split="dev")
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)
orig_cfg = copy.deepcopy(model.cfg.decoding)
real_cls = rd.GPUBoostingTreeModel


class TrieShim:
    """Stands in for GPUBoostingTreeModel: same from_config call, your trie inside."""

    bonus = 1.0
    calls = 0

    @classmethod
    def from_config(cls, cfg, tokenizer=None):
        cls.calls += 1
        tok = tokenizer if tokenizer is not None else model.tokenizer
        phrases = getattr(cfg, "key_phrases_list", None)
        if phrases is None:
            phrases = cfg["key_phrases_list"]
        return TrieFusionModel(list(phrases), tok, tok.vocab_size, bonus=cls.bonus)


def best_text(out):
    first = out[0]
    best = first[0] if isinstance(first, list) else first
    return best.text


def decode(audio, phrases, own, bonus=1.0, alpha=1.0):
    cfg = copy.deepcopy(orig_cfg)
    with open_dict(cfg):
        cfg.strategy = "malsd_batch"
        cfg.beam.beam_size = 4
        cfg.beam.allow_cuda_graphs = False
        cfg.beam.boosting_tree = {
            "key_phrases_list": phrases,
            "context_score": 1.0,
            "depth_scaling": 2.0,
        }
        cfg.beam.boosting_tree_alpha = alpha
    TrieShim.bonus = bonus
    rd.GPUBoostingTreeModel = TrieShim if own else real_cls
    model.change_decoding_strategy(cfg)
    return best_text(model.transcribe([audio]))


for idx in CLIPS:
    s = dataset[idx]
    audio = s["audio"].get_all_samples().data.squeeze()
    phrases = list(s["dictionary"])
    print(f"RESULT ===== dev clip {idx}  names: {phrases}")
    rd.GPUBoostingTreeModel = real_cls
    model.change_decoding_strategy(orig_cfg)
    print("RESULT baseline    :", best_text(model.transcribe([audio])))
    print("RESULT NeMo a=2.0  :", decode(audio, phrases, own=False, alpha=2.0))
    for b in BONUSES:
        print(f"RESULT own bonus={b}:", decode(audio, phrases, own=True, bonus=b))

print("RESULT shim was built", TrieShim.calls, "times")
