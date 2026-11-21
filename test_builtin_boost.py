from datasets import load_dataset
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

INDEX = 0  # change this to try another clip

dataset = load_dataset("argmaxinc/contextual-earnings22", split="test")
sample = dataset[INDEX]

audio = sample["audio"].get_all_samples()
audio_tensor = audio.data.squeeze()

phrases = list(sample["dictionary"])

print("GROUND TRUTH:", sample["text"])
print("KEYWORDS:", sample["keywords"])
print("DICTIONARY:", phrases)

model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

print("\nBASELINE:", model.transcribe([audio_tensor])[0].text)

cfg = model.cfg.decoding
with open_dict(cfg):
    cfg.strategy = "malsd_batch"
    cfg.beam.beam_size = 4
    cfg.beam.allow_cuda_graphs = False
    cfg.beam.boosting_tree = {
        "key_phrases_list": phrases,
        "context_score": 1.0,
        "depth_scaling": 2.0,
    }

for beam in [4, 8, 16]:
    for alpha in [1.5, 2.0, 2.5, 3.0]:
        with open_dict(cfg):
            cfg.beam.beam_size = beam
            cfg.beam.boosting_tree_alpha = alpha
        model.change_decoding_strategy(cfg)
        out = model.transcribe([audio_tensor])[0]
        best = out[0] if isinstance(out, list) else out
        print(f"\nBEAM {beam} ALPHA {alpha}: {best.text}")
