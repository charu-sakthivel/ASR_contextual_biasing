from collections import deque

import torch
from torch import nn
import math


class TrieFusionModel(nn.Module):
    """Table-based, batched trie for NeMo's fusion_models interface.

    State = trie node id (0 = root). The bonus for the token at depth d is
    bonus * d**gamma. Each state has a potential = total bonus given so far
    for its partial match; a step adds potential(new) - potential(old), so
    leaving the trie revokes the pending bonus. With fallback=True, a broken
    match falls back to the longest suffix that is still a prefix of some
    phrase (Aho-Corasick fail arcs) instead of resetting to the root.
    """

    def __init__(
        self,
        phrases,
        tokenizer,
        vocab_size,
        bonus=1.0,
        gamma=0.0,
        fallback=False,
        schedule="power",
    ):
        super().__init__()
        children, is_end, depth = [{}], [False], [0]
        for p in phrases:
            for variant in {p, p.lower(), p.title()}:
                node = 0
                for tid in tokenizer.text_to_ids(variant):
                    nxt = children[node].get(tid)
                    if nxt is None:
                        children.append({})
                        is_end.append(False)
                        depth.append(depth[node] + 1)
                        nxt = len(children) - 1
                        children[node][tid] = nxt
                    node = nxt
                is_end[node] = True

        def tok_bonus(d):
            if schedule == "nemo":
                return bonus if d == 1 else 2.0 * bonus + math.log(d)
            return bonus * (d**gamma)

        cum = [0.0]
        for d in range(1, max(depth) + 1):
            cum.append(cum[-1] + tok_bonus(d))

        n = len(children)
        fail = [0] * n
        queue = deque(children[0].values())
        while queue:
            s = queue.popleft()
            for v, c in children[s].items():
                f = fail[s]
                while f != 0 and v not in children[f]:
                    f = fail[f]
                cand = children[f].get(v, 0)
                fail[c] = cand if cand != c else 0
                queue.append(c)

        def goto(s, v):
            if fallback:
                while s != 0 and v not in children[s]:
                    s = fail[s]
                return children[s].get(v, 0)
            if v in children[s]:
                return children[s][v]
            return children[0].get(v, 0)

        pot = [0.0 if is_end[s] else cum[depth[s]] for s in range(n)]
        trie_tokens = sorted({v for ch in children for v in ch})

        scores = torch.zeros(n, vocab_size)
        next_states = torch.zeros(n, vocab_size, dtype=torch.long)
        for s in range(n):
            scores[s, :] = -pot[s]
            for v in trie_tokens:
                t = goto(s, v)
                scores[s, v] = cum[depth[t]] - pot[s]
                next_states[s, v] = 0 if is_end[t] else t

        self.register_buffer("score_table", scores)
        self.register_buffer("next_table", next_states)
        self.num_nodes = n

    def get_init_states(self, batch_size, bos=True):
        return torch.zeros(batch_size, dtype=torch.long, device=self.score_table.device)

    def advance(self, states, eos_id=None):
        return self.score_table[states], self.next_table[states]
