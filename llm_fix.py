"""Can a small language model recover names the recognizer never wrote?

Reads saved predictions (preds_<split>.json), applies the fuzzy correction,
then asks a local language model (via Ollama) to propose replacements for
names on the clip's list that are still missing. Only safe, checkable
replacements are applied: the "wrong" words must appear in the transcript,
be 1-3 words long, and the name must come from the clip's own list.

Usage (Ollama must be running, with the model pulled first):
    python llm_fix.py --split dev --model qwen2.5:3b
    python llm_fix.py --split dev --model qwen2.5:3b --limit 10   # quick try

Model answers are cached, so re-running is cheap.
"""

import argparse
import hashlib
import json
import os
import random
import re
import urllib.request

import jiwer
from jiwer import wer

from context_biasing.text_utils import fuzzy_fix, norm, present

OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """You are checking a speech recognition transcript of a business call.
These names may have been spoken in the recording but do not appear in the transcript:
{names}

Transcript:
{text}

Some of these names may have been misheard and written as different words
(for example, "Lumora" written as "Lou Mora").
For each name that you are confident was spoken, copy the exact misheard words
from the transcript (one to three words).
The "name" must be copied exactly as it is written in the list above, character
for character. Do not add, remove or change any words in it.
Do not guess. If you are not confident about a name, leave it out.
Answer only with JSON in this form:
{{"replacements": [{{"wrong": "words copied from the transcript", "name": "name from the list"}}]}}"""


def ask(model, prompt):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]


def propose(model, text, missing):
    prompt = PROMPT.format(names="\n".join(f"- {n}" for n in missing), text=text)
    try:
        out = json.loads(ask(model, prompt))
        reps = out.get("replacements", []) if isinstance(out, dict) else []
        return reps if isinstance(reps, list) else []
    except Exception as e:  # bad JSON or a connection problem
        print(f"  model error: {e}")
        return []


def apply(text, reps, missing, all_names):
    """Apply only replacements that pass every safety check."""
    lookup = {n.lower(): n for n in missing}
    applied = []
    for r in reps:
        if not isinstance(r, dict):
            continue
        wrong = str(r.get("wrong", "")).strip()
        name = lookup.get(str(r.get("name", "")).strip().lower())
        if not wrong or not name or not norm(wrong) or len(wrong.split()) > 3:
            continue
        if norm(wrong) == norm(name):
            continue
        if any(present(wrong, n) for n in all_names):  # never overwrite a listed name
            continue
        pat = re.compile(r"(?<!\w)" + re.escape(wrong) + r"(?!\w)", re.IGNORECASE)
        if not pat.search(text):  # the model must quote the transcript exactly
            continue
        text = pat.sub(name, text, count=1)
        applied.append((wrong, name))
        del lookup[name.lower()]
    return text, applied


def stats(preds, refs, dicts):
    hits = total = false_names = 0
    for p, r, d in zip(preds, refs, dicts):
        for n in d:
            in_ref, in_pred = present(r, n), present(p, n)
            total += in_ref
            hits += in_ref and in_pred
            false_names += in_pred and not in_ref
    w = wer([norm(r) for r in refs], [norm(p) for p in preds])
    return hits, total, false_names, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument(
        "--model", required=True, help="an Ollama model name, e.g. qwen2.5:3b"
    )
    ap.add_argument(
        "--base-key",
        default="boosted_2.0",
        help="which saved predictions to start from",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="only the first N clips (quick test)"
    )
    ap.add_argument(
        "--bootstrap",
        action="store_true",
        help="also print a paired bootstrap CI vs. the starting predictions",
    )
    args = ap.parse_args()

    d = json.load(open(f"preds_{args.split}.json"))
    refs, dicts = d["refs"], d["dicts"]
    base = [fuzzy_fix(p, n) for p, n in zip(d[args.base_key], dicts)]
    if args.limit:
        refs, dicts, base = refs[: args.limit], dicts[: args.limit], base[: args.limit]

    tag = re.sub(r"[^\w.-]", "_", args.model)
    # The prompt's fingerprint is part of the cache name, so changing the prompt
    # never reuses answers given to an older prompt.
    prompt_id = hashlib.sha1(PROMPT.encode()).hexdigest()[:8]
    cache_path = f"llm_cache_{args.split}_{tag}_{prompt_id}.json"
    print(f"prompt version {prompt_id}, cache {cache_path}")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    out, swaps = [], []
    for i, (text, names) in enumerate(zip(base, dicts)):
        missing = [n for n in names if not present(text, n)]
        if not missing:
            out.append(text)
            continue
        key = str(i)
        if key not in cache:
            print(f"clip {i + 1}/{len(base)}: asking about {len(missing)} names")
            cache[key] = propose(args.model, text, missing)
            json.dump(cache, open(cache_path, "w"))
        new, applied = apply(text, cache[key], missing, names)
        out.append(new)
        for wrong, name in applied:
            swaps.append((i, wrong, name, present(refs[i], name)))

    for label, preds in [("fuzzy (start)", base), ("fuzzy + LLM", out)]:
        h, t, f, w = stats(preds, refs, dicts)
        print(
            f"RESULT {label:14s} recall {h}/{t} ({h / t:.3f})  WER {w:.4f}  names not in reference: {f}"
        )

    good = sum(s[3] for s in swaps)
    print(
        f"RESULT LLM swaps: {len(swaps)}, name in reference: {good}, not in reference: {len(swaps) - good}"
    )
    for i, wrong, name, ok in swaps:
        print(f"RESULT   clip {i}: {wrong} -> {name} | in reference: {ok}")

    json.dump(
        {"base": base, "llm": out, "swaps": swaps, "prompt_id": prompt_id},
        open(f"preds_llm_{args.split}_{tag}_{prompt_id}.json", "w"),
    )

    if args.bootstrap:
        lo_r, hi_r, lo_w, hi_w = paired_bootstrap(base, out, refs, dicts)
        print(
            f"RESULT fuzzy+LLM minus fuzzy (95% CI): recall [{lo_r:+.4f}, {hi_r:+.4f}]  WER [{lo_w:+.4f}, {hi_w:+.4f}]"
        )


def paired_bootstrap(a_preds, b_preds, refs, dicts, n_boot=2000, seed=0):
    """Paired bootstrap over clips: differences b - a in name recall and WER."""
    rows = []
    for a, b, r, d in zip(a_preds, b_preds, refs, dicts):
        ref = norm(r)
        if not ref:
            continue
        nref = len(ref.split())
        tot = sum(1 for n in d if present(r, n))
        row = [nref, tot]
        for p in (a, b):
            hyp = norm(p)
            if hyp:
                o = jiwer.process_words(ref, hyp)
                errs = o.substitutions + o.deletions + o.insertions
            else:
                errs = nref
            hits = sum(1 for n in d if present(r, n) and present(p, n))
            row += [errs, hits]
        rows.append(row)

    rng = random.Random(seed)
    d_rec, d_wer = [], []
    for _ in range(n_boot):
        s = [rows[rng.randrange(len(rows))] for _ in rows]
        words = sum(x[0] for x in s)
        names = sum(x[1] for x in s) or 1
        d_wer.append((sum(x[4] for x in s) - sum(x[2] for x in s)) / words)
        d_rec.append((sum(x[5] for x in s) - sum(x[3] for x in s)) / names)
    d_rec.sort()
    d_wer.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return d_rec[lo], d_rec[hi], d_wer[lo], d_wer[hi]


if __name__ == "__main__":
    main()
