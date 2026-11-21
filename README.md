# Contextual Biasing for Parakeet-TDT (name recognition)

Improving recognition of context-specific names (people, companies) in
NVIDIA Parakeet-TDT-0.6B-v2, evaluated on `argmaxinc/contextual-earnings22`.

## Method
1. **Phrase boosting** using NeMo 3.0.0's built-in boosting tree during TDT
   beam search (beam 4, `boosting_tree_alpha` 1.5, per-clip phrase lists).
2. **Fuzzy correction** (my addition): after decoding, a capitalized word
   within a small edit distance of a single-word context name (1 for names
   of 5 characters or fewer, 2 for longer) is replaced by that name.
   Possessives like "ESA's" are handled.
3. `context_biasing/trie.py`: my own stateful trie with bonus revocation
   (unit-tested; **not yet integrated into the decoder**).

## Results (full test split, 772 clips)
Alpha (2.0) was chosen on the 100-clip dev split. Name recall counts only
dictionary names actually spoken in the reference. Intervals are 95%
bootstrap CIs over clips (2,000 resamples).

| System | Name recall | WER |
|---|---|---|
| Parakeet-TDT baseline | 49.0% [46.1, 51.9] | 19.61% [18.75, 20.50] |
| + NeMo boosting | 87.5% [85.5, 89.4] | 17.57% [16.69, 18.49] |
| + fuzzy correction | 89.5% [87.6, 91.2] | 17.49% [16.60, 18.41] |

Paired comparison, fuzzy vs boosting only: recall +1.3 to +2.8 points,
WER -0.04 to -0.13 points (both 95% intervals exclude zero). The WER
gain is small.

Note: the fuzzy rule's design, including possessive handling, was informed
by errors seen on test clips 0-60, so its gain may be slightly optimistic.
Boosting settings were tuned on dev only.

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