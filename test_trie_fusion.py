import random

import nemo.collections.asr as nemo_asr

from context_biasing.trie import TrieBiasing
from context_biasing.trie_fusion import TrieFusionModel

model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)
tok = model.tokenizer
V = tok.vocab_size
names = ["Pryanova", "Gazprom", "Anna", "Tele2 Disk"]

ref = TrieBiasing(names, tok, bonus=1.0)
fus = TrieFusionModel(names, tok, V, bonus=1.0)

states = fus.get_init_states(batch_size=3)
scores, nxt = fus.advance(states)
print(
    f"RESULT vocab={V} nodes={fus.num_nodes} scores {tuple(scores.shape)} {scores.dtype}, next {tuple(nxt.shape)} {nxt.dtype}"
)

pool = sorted(
    {t for n in names for v in (n, n.lower(), n.title()) for t in tok.text_to_ids(v)}
)
random.seed(0)
mismatches = 0
steps = 0
for _ in range(2000):
    r_state = ref.init_state()
    f_state = fus.get_init_states(1)
    for _ in range(30):
        t = random.choice(pool) if random.random() < 0.7 else random.randrange(V)
        r_delta, r_state = ref.step(r_state, t)
        f_scores, f_next = fus.advance(f_state)
        f_delta = float(f_scores[0, t])
        f_state = f_next[:, t]
        steps += 1
        if abs(r_delta - f_delta) > 1e-6:
            mismatches += 1
print(f"RESULT compared {steps} steps, mismatches: {mismatches}")
