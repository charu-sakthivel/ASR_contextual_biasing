class TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children = {}
        self.is_end = False


class TrieBiasing:
    def __init__(self, phrases, tokenizer, bonus=2.0):
        self.root = TrieNode()
        self.bonus = bonus
        for p in phrases:
            for variant in {p, p.lower(), p.title()}:
                node = self.root
                for tid in tokenizer.text_to_ids(variant):
                    node = node.children.setdefault(tid, TrieNode())
                node.is_end = True

    def init_state(self):
        return (self.root, 0.0)  # (current node, pending bonus)

    def step(self, state, token_id):
        """Return (score_delta, new_state) for emitting token_id."""
        node, pending = state
        if token_id in node.children:
            nxt = node.children[token_id]
            if nxt.is_end:
                return self.bonus, (self.root, 0.0)  # full match: keep bonus
            return self.bonus, (nxt, pending + self.bonus)
        delta = -pending  # left the trie: revoke bonus
        if token_id in self.root.children:  # may start a new phrase
            nxt = self.root.children[token_id]
            return delta + self.bonus, (nxt, self.bonus)
        return delta, (self.root, 0.0)
