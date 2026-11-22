import torch
from torch import nn


class TrieFusionModel(nn.Module):
    """Table-based, batched trie for NeMo's fusion_models interface.

    State = trie node id (0 = root). The bonus for the token at depth d is
    bonus * d**gamma (gamma=0 gives the flat bonus tested earlier). Leaving
    the trie revokes everything given so far for the partial match.
    """

    def __init__(self, phrases, tokenizer, vocab_size, bonus=1.0, gamma=0.0):
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
            return bonus * (d**gamma)

        cum = [0.0]
        for d in range(1, max(depth) + 1):
            cum.append(cum[-1] + tok_bonus(d))

        n = len(children)
        scores = torch.zeros(n, vocab_size)
        next_states = torch.zeros(n, vocab_size, dtype=torch.long)
        for s in range(n):
            leave = -cum[depth[s]]
            scores[s, :] = leave
            for v, c in children[0].items():
                scores[s, v] = leave + tok_bonus(1)
                next_states[s, v] = 0 if is_end[c] else c
            for v, c in children[s].items():
                scores[s, v] = tok_bonus(depth[c])
                next_states[s, v] = 0 if is_end[c] else c

        self.register_buffer("score_table", scores)
        self.register_buffer("next_table", next_states)
        self.num_nodes = n

    def get_init_states(self, batch_size, bos=True):
        return torch.zeros(batch_size, dtype=torch.long, device=self.score_table.device)

    def advance(self, states, eos_id=None):
        return self.score_table[states], self.next_table[states]
