# Contextual Biasing for Parakeet-TDT: Recognizing Rare Names

Speech recognizers handle everyday speech well but often get rare names
wrong. On the **contextual-earnings22** benchmark (earnings-call audio, where
each clip comes with a short list of names that may appear in it), NVIDIA's
**Parakeet-TDT-0.6B-v2** writes only about half of the spoken names
correctly. This project studies how to use each clip's name list to fix
that, on a CPU-only laptop (MacBook Air, NeMo 3.0.0, Python 3.11).

## Methods

**1. NeMo phrase boosting (baseline method).** NeMo's built-in boosting tree,
used inside TDT beam search (beam size 4, fusion weight alpha = 2.0, chosen on
the dev split).

**2. My trie decoder** (`context_biasing/trie_fusion.py`). The phrase-boosting
idea described in Andrusenko et al., "TurboBias" (arXiv:2508.07014),
re-implemented from scratch in Python with no GPU code. Every name on the
clip's list is split into the model's word pieces and stored in a trie. While
the beam search decodes, each hypothesis tracks its place in the trie: a word
piece that continues a name earns a bonus, and if the match breaks off, the
bonus it earned is taken back. For each clip, the trie is precomputed into two
lookup tables (score change and next state, for every state and word piece),
and it plugs into NeMo's TDT beam search through the same `fusion_models`
interface that NeMo uses for its own tree, so the model and decoder are
otherwise unchanged. It was checked against a plain, one-token-at-a-time
reference trie (`context_biasing/trie.py`): 0 mismatches over 60,000 random
steps.

The reward schedule matters. My first schedule (a smooth power curve in
match depth) trailed NeMo. The final version gives the first word piece of a
name a small reward (1) and each later piece a larger one (2 plus the natural
log of its position: about 2.7 for the second piece, 3.1 for the third),
following the schedule described in the paper; all scores are multiplied by a
fusion weight of 2.0.

**3. Fuzzy correction** (`context_biasing/text_utils.py`). A model-agnostic
step that runs after decoding: a capitalized word within a small edit
distance of a single-word name on the clip's list (1 edit for names of 5
characters or fewer, 2 for longer names) is replaced by that name, for
example "Crip.com" → "Trip.com". Possessives are handled.

**4. Language-model recovery (exploratory)** (`llm_fix.py`). A small local
language model (qwen2.5:3b via Ollama) is given the transcript and the names
that are still missing, and proposes replacements of the form
"misheard words → name". A replacement is applied only if the name comes from
the clip's list, the quoted words (1 to 3) appear exactly in the transcript,
and no correctly written listed name is overwritten.

## Results (full test split, 772 clips)

Name recall counts dictionary names that are spoken in the reference and also
appear in the output. Intervals are 95% bootstrap confidence intervals over
clips (2,000 resamples). All settings were chosen on the 100-clip dev split.

| System | Name recall | WER |
|---|---|---|
| Parakeet-TDT baseline | 49.0% [46.1, 51.9] | 19.61% [18.75, 20.50] |
| + fuzzy correction only | 64.8% [62.0, 67.7] | 18.81% [17.94, 19.72] |
| + NeMo boosting | 87.5% [85.5, 89.4] | 17.57% [16.69, 18.49] |
| + NeMo boosting + fuzzy | 89.4% [87.6, 91.2] | 17.49% [16.60, 18.41] |
| + my trie (first schedule) | 83.6% [81.4, 85.8] | 17.65% [16.78, 18.60] |
| + my trie (final schedule) | 87.8% [85.8, 89.7] | 17.56% [16.69, 18.54] |
| + my trie (final schedule) + fuzzy | 89.7% [87.9, 91.4] | 17.48% [16.60, 18.46] |

**Paired comparisons (95% CI of the difference):**

| Comparison | Name recall | WER |
|---|---|---|
| Fuzzy only vs. baseline | +13.6 to +18.0 points | −0.93 to −0.68 points |
| NeMo boosting vs. baseline | +35.6 to +41.4 points | −2.43 to −1.68 points |
| NeMo + fuzzy vs. NeMo | +1.3 to +2.7 points | −0.12 to −0.04 points |
| My trie (final) vs. NeMo | −0.2 to +0.9 points | −0.15 to +0.20 points |
| My trie (final) + fuzzy vs. NeMo + fuzzy | −0.3 to +0.9 points | −0.15 to +0.19 points |

**Summary.**
- Fuzzy correction improves name recall on its own (49.0% → 64.8%) without
  touching the decoder, and adds a further 1.3 to 2.7 points on top of NeMo's
  boosting. Both intervals exclude zero.
- My trie with the final schedule scored slightly higher than NeMo's boosting
  in this evaluation (87.8% vs. 87.5% recall, 17.56% vs. 17.57% WER), but the
  differences are within the confidence intervals, so the two perform the
  same. With the first schedule, it trailed NeMo by 2.5 to 5 points of recall.

### Language-model recovery (exploratory)

Starting from NeMo boosting + fuzzy correction, one run on the test split:

| | Name recall | WER |
|---|---|---|
| Before (NeMo + fuzzy) | 89.4% (1162/1300) | 17.49% |
| After language-model recovery | 91.6% (1191/1300) | 17.42% |
| Paired difference (95% CI) | +1.5 to +3.1 points* | −0.12 to −0.01 points |

The model changed 29 clips. Comparing each clip's word errors before and after
(`check_swaps.py`): **21 improved, 4 got worse, 4 were unchanged.** The 4 that
got worse replaced correct words with a name that was spoken elsewhere in the
clip (for example, "flavored" → "Qinfeng").

*The recall metric checks only that a name appears somewhere in the clip, not
at the right position, so it counts those 4 as hits; the recall gain is an
upper bound. WER does check positions, and its improvement is small but its
interval excludes zero. A guard against misplaced names (for example,
requiring the replaced words to resemble the name) is the next step and has
not been tested.

## Analysis: why do names still get missed?

On the test split, the first version of my trie (without fuzzy correction)
missed 213 spoken names; 149 of them were also missed by NeMo's boosting, so
most remaining errors are not caused by the trie.

| Type of miss (213 names) | Share |
|---|---|
| No trace of the name in the output | 46% |
| Near-miss (spelling variant) | 25% |
| Multi-word or punctuated name* | 16% |
| Start right, end wrong | 7% |
| Start wrong, end right | 6% |

*Partly an artifact of my text normalizer, which treats "." as a separator.
Categories come from simple text-similarity rules, so treat them as
approximate. This analysis was run before the final schedule; the shares may
differ for the final system.

In an n-best check on four of these misses, only one (Bayon) had the correct
name anywhere in the model's top candidates. When the acoustic model never
proposes a name, decoder-side biasing has nothing to boost.

**Explanations tested for the gap between my first trie and NeMo:**
- Matching from the middle of a name: not supported (only 6% of misses had a
  wrong start and correct end).
- Name length: the gap was similar for names of 2 to 6+ word pieces.
- Fallback after a broken match (Aho-Corasick fail arcs): the lookup tables
  were identical for 0 of 100 dev clips' name lists, so results did not change.
- NeMo's blank-token weighting: zero-bonus controls showed that the weighting
  alone does not improve recall or WER, and running my trie with NeMo's fusion
  weight gave the same test result (recall 83.7%, WER 17.63%).
- **Cause found:** the reward schedule. Switching to the schedule described in
  the TurboBias paper closed the gap.

**Other ideas tested, with no measurable effect:**
- Phonetic (Metaphone) correction on top of fuzzy correction: no change in
  recall or WER on the test split.
- NeMo's word-piece variant option (`use_bpe_dropout`, 5 variants): recall
  +0.3 points (CI −0.3 to +1.0), WER +0.05 points (CI −0.06 to +0.17); recall
  also varied by 2 names across identical dev runs.
- Names that are prefixes of other names: affects only 4 of 1,300 spoken
  names, so it was not pursued.

## Caveats

- My trie was run on the test split in three configurations, in this order,
  each after dev results: first schedule with fusion weight 1.0; first
  schedule with fusion weight 2.0; final schedule with fusion weight 2.0. The
  last one's settings were fixed on dev before its single test run. My trie had
  more settings tried on dev (about 20) than NeMo's fusion weight (4).
- The fuzzy rule's thresholds were fixed before evaluation, but its possessive
  handling was added after seeing an error in test clip 55. A bug with
  possessive dictionary entries was fixed before the final numbers were
  computed. On fuzzy correction alone, 219 of 225 swaps put a name that
  appears in the clip's reference; this checks presence, not position.
- The language-model prompt was adjusted once after a dev run, to require
  names copied exactly from the list. The test split was then run once.
- Fuzzy correction only handles single-word names.
- Results are from one dataset and one ASR model.

## Files

- `context_biasing/trie.py`: plain, one-token-at-a-time reference trie.
- `context_biasing/trie_fusion.py`: table-based trie used inside the beam search.
- `context_biasing/text_utils.py`: text normalization and fuzzy correction.
- `evaluate.py`: baseline and NeMo boosting (dev sweep or one test run).
- `evaluate_own.py`, `evaluate_own_test.py`, `evaluate_sched_dev.py`,
  `evaluate_sched_test.py`: my trie on dev and test.
- `bootstrap.py`, `bootstrap_compare.py`, `recompute_final.py`: confidence
  intervals and paired comparisons.
- `llm_fix.py`, `check_swaps.py`: language-model recovery and the per-clip
  check of its swaps.
- Analysis and negative-result scripts: `failure_analysis.py`,
  `failure_by_length.py`, `check_fallback.py`, `check_prefix_overlap.py`,
  `phonetic_dev.py`, `phonetic_test.py`, `bpe_dev.py`, `evaluate_bpe_test.py`,
  `fuzzy_alone_test.py`.

## Reproducing

Prediction files (`preds_*.json`) are not in the repository. Generate them
first; the other scripts read them.

```
python evaluate.py --split dev                 # tunes alpha on dev
python evaluate.py --split test --alphas 2.0   # writes preds_test.json
python evaluate_sched_test.py                  # my trie, final schedule
python recompute_final.py                      # final tables and intervals
python llm_fix.py --split test --model qwen2.5:3b --bootstrap   # needs Ollama
```

CPU only. A full test-split pass takes roughly 20 to 30 minutes on a MacBook Air.

## References

- A. Andrusenko, V. Bataev, L. Grigoryan, V. Lavrukhin, B. Ginsburg.
  "TurboBias: Universal ASR Context-Biasing powered by GPU-accelerated
  Phrase-Boosting Tree." arXiv:2508.07014, 2025.
- H. Xu, F. Jia, S. Majumdar, H. Huang, S. Watanabe, B. Ginsburg. "Efficient
  Sequence Transduction by Jointly Predicting Tokens and Durations." ICML 2023
  (arXiv:2304.06795).
- NVIDIA NeMo toolkit: https://github.com/NVIDIA/NeMo
- Model: `nvidia/parakeet-tdt-0.6b-v2`
- Dataset: `argmaxinc/contextual-earnings22`
