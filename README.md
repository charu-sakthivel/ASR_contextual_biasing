# Contextual Biasing for Parakeet-TDT (name recognition)

Improving recognition of context-specific names (people, companies) in
NVIDIA Parakeet-TDT-0.6B-v2, evaluated on `argmaxinc/contextual-earnings22`.

## Method
1. **NeMo phrase boosting** (NeMo 3.0.0 boosting tree, beam 4, alpha 2.0
   chosen on the dev split).
2. **My own trie decoder** (`context_biasing/trie_fusion.py`): a
   table-based, batched trie with depth-scaled bonuses and revocation of
   the bonus when a partial match breaks. It is plugged into NeMo's TDT
   beam search through the `fusion_models` interface, replacing NeMo's
   boosting tree. Checked against a plain-Python reference trie (0
   mismatches over 60,000 random steps). Bonus 2.0, depth exponent 0.5,
   chosen on dev.
3. **Fuzzy correction** (`context_biasing/text_utils.py`): after decoding,
   a capitalized word within a small edit distance of a single-word context
   name (1 for names of 5 characters or fewer, 2 for longer) is replaced
   by that name. Possessives are handled.

## Results (full test split, 772 clips)
Name recall counts only dictionary names actually spoken in the reference.
Intervals are 95% bootstrap CIs over clips (2,000 resamples). All settings
were chosen on the 100-clip dev split; the test split was run once per system.

| System | Name recall | WER |
|---|---|---|
| Parakeet-TDT baseline | 49.0% [46.1, 51.9] | 19.61% [18.75, 20.50] |
| + NeMo boosting | 87.5% [85.5, 89.4] | 17.57% [16.69, 18.49] |
| + my trie | 83.6% [81.4, 85.8] | 17.65% [16.78, 18.60] |
| + NeMo boosting + fuzzy | 89.5% [87.6, 91.2] | 17.49% [16.60, 18.41] |
| + my trie + fuzzy | 87.2% [85.2, 89.1] | 17.49% [16.62, 18.42] |

Paired comparison, my trie minus NeMo boosting: recall -5.3 to -2.5 points
(-3.6 to -1.1 with fuzzy), so NeMo is reliably better on name recall.
WER differences are within noise in both cases (intervals include zero).

Caveats: the fuzzy rule's design was informed by errors seen on test clips
0-60. My trie had more settings tried on dev (13) than NeMo's alpha (4).

## Failure cases
- Hepsipapel, Trip.com, ISA: the model's first guess is acoustically
  wrong, so boosting alone cannot recover them (fuzzy correction fixes
  some of these).
- "Bayon" loses to the common word "Bayonne" at default settings.
- Names are only corrected when single-word; multi-word names are not
  handled by the fuzzy step.

## Setup
CPU only (MacBook Air), NeMo 3.0.0, Python 3.11.
Run: `python test_fuzzy.py`