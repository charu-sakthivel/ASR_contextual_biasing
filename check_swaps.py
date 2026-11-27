"""Did each LLM swap land at the right place in the transcript?

"Name in reference" only checks that the name appears somewhere in the clip's
reference, so a swap can count as a hit even if it replaced the wrong words.
This script compares each swapped clip's word errors before and after the swap:
  fewer errors  -> the swap fixed the transcript (right place)
  more errors   -> it replaced words that were actually correct (wrong place)
  same errors   -> unclear; check by eye

Usage:
    python check_swaps.py --split test --file preds_llm_test_qwen2.5_3b_7f26e8e6.json
"""

import argparse
import json

import jiwer

from context_biasing.text_utils import norm


def errors(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    if not hyp:
        return len(ref.split())
    o = jiwer.process_words(ref, hyp)
    return o.substitutions + o.deletions + o.insertions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--file", required=True)
    args = ap.parse_args()

    refs = json.load(open(f"preds_{args.split}.json"))["refs"]
    d = json.load(open(args.file))
    base, llm, swaps = d["base"], d["llm"], d["swaps"]

    by_clip = {}
    for i, wrong, name, _ in swaps:
        by_clip.setdefault(i, []).append(f"{wrong} -> {name}")

    better = worse = same = 0
    for i, desc in sorted(by_clip.items()):
        before, after = errors(refs[i], base[i]), errors(refs[i], llm[i])
        if after < before:
            verdict, better = "FIXED", better + 1
        elif after > before:
            verdict, worse = "WORSE", worse + 1
        else:
            verdict, same = "SAME", same + 1
        print(
            f"RESULT clip {i}: {'; '.join(desc)} | errors {before} -> {after} | {verdict}"
        )

    print(
        f"RESULT clips with swaps: {len(by_clip)}, fixed: {better}, worse: {worse}, unchanged: {same}"
    )


if __name__ == "__main__":
    main()
