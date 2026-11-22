import re


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def present(text, ph):
    return " " + norm(ph) + " " in " " + norm(text) + " "


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fuzzy_fix(pred, phrases):
    names = [p for p in phrases if " " not in p]
    out = []
    for w in pred.split(" "):
        pre, core, post = re.match(r"^(\W*)(.*?)((?:['’]s)?\W*)$", w).groups()
        if core and core[0].isupper():
            best = None
            for n in names:
                if core.lower() == n.lower():
                    best = None
                    break
                d = lev(core.lower(), n.lower())
                limit = 1 if len(n) <= 5 else 2
                if d <= limit and (best is None or d < best[0]):
                    best = (d, n)
            if best:
                core = best[1]
        out.append(pre + core + post)
    return " ".join(out)
