"""Recompute the headline results of the SORA-LLM study from the minimal dataset.

Usage:  python3 check_key_results.py            (Python >= 3.8, standard library only)
Prints: the primary outcome table (Table 2 in the article), the sources of wrong SAILs (Figure 2),
extraction accuracy, repeatability (Fleiss' kappa) and cost per correct assessment.
The expected output is in expected_output.txt.
"""
import csv
import math
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["claude-haiku-4-5", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
          "gpt-oss:20b", "mistral-small3.2:24b", "gemma4:31b", "qwen3.8:27b"]
CONDS = ["closed_book", "reference", "extract"]
LABEL = {"closed_book": "Closed book", "reference": "Rule card", "extract": "Extract + calculator"}


def read(name):
    with open(os.path.join(HERE, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def fleiss_kappa(table):
    n_items, n_raters = len(table), sum(table[0])
    p_j = [sum(row[j] for row in table) / (n_items * n_raters) for j in range(len(table[0]))]
    p_i = [(sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1)) for row in table]
    p_bar, p_e = sum(p_i) / n_items, sum(p * p for p in p_j)
    return (p_bar - p_e) / (1 - p_e)


A = read("assessments.csv")
print(f"assessments: {len(A)} rows; scenarios: {len(read('scenarios.csv'))}; "
      f"repeatability: {len(read('repeatability_haiku55.csv'))}; extractions: {len(read('extractions.csv'))}\n")

# 1. Primary outcome -------------------------------------------------------------------------------
print("1. SAIL under-classification (primary) and exact SAIL, Wilson 95% CI")
print(f"{'model':22s} {'set-up':21s} {'lang':4s} {'n':>5s}  {'SAIL too low':22s} {'SAIL exact':22s} too low/too high")
groups = defaultdict(list)
for r in A:
    groups[(r["model"], r["condition"], r["lang"])].append(r)
for m in MODELS:
    for c in CONDS:
        for lang in ("en", "tr"):
            g = groups[(m, c, lang)]
            n = len(g)
            under = sum(r["sail_under"] == "1" for r in g)
            over = sum(r["sail_over"] == "1" for r in g)
            exact = sum(r["ok_sail"] == "1" for r in g)
            print(f"{m:22s} {LABEL[c]:21s} {lang:4s} {n:5d}  {pct(under, n):22s} {pct(exact, n):22s} {under}/{over}")

# 2. Sources of wrong SAILs --------------------------------------------------------------------------
print("\n2. Share of assessments with a wrong SAIL by source (% of all assessments; both languages)")
srcs = ["ground", "air", "ground+air", "lookup", "unparsed"]
print(f"{'model':22s} {'set-up':21s} " + " ".join(f"{s:>10s}" for s in srcs))
for m in MODELS:
    for c in CONDS:
        g = groups[(m, c, "en")] + groups[(m, c, "tr")]
        cnt = Counter(r["sail_err_source"] for r in g)
        print(f"{m:22s} {LABEL[c]:21s} " + " ".join(f"{100 * cnt[s] / len(g):10.1f}" for s in srcs))

# 3. Extraction accuracy -------------------------------------------------------------------------------
print("\n3. Extract + calculator: accuracy of each extracted field (% of returned extractions; both languages)")
fields = ["ex_mtom", "ex_dim", "ex_speed", "ex_band", "ex_mitigations", "ex_height_side", "ex_airspace_relevant",
          "ex_flightmode"]
print(f"{'model':22s} {'returned':>9s} " + " ".join(f"{f[3:]:>17s}" for f in fields))
for m in MODELS:
    g = [r for r in groups[(m, "extract", "en")] + groups[(m, "extract", "tr")] if r["ex_parsed"] == "1"]
    print(f"{m:22s} {len(g):9d} " + " ".join(f"{100 * sum(r[f] == '1' for r in g) / len(g):17.1f}" for f in fields))

# 4. Repeatability -------------------------------------------------------------------------------------
cats = ["I", "II", "III", "IV", "V", "VI", "certified", "missing"]
by_sid = defaultdict(list)
for r in read("repeatability_haiku55.csv"):
    by_sid[r["sid"]].append(r["pred_sail"] if r["pred_sail"] in cats else "missing")
table = [[v.count(c) for c in cats] for v in by_sid.values()]
same = sum(len(set(v)) == 1 for v in by_sid.values())
print(f"\n4. Repeatability (Haiku 5.5, closed book, English, {len(by_sid)} scenarios x 3 runs): "
      f"Fleiss kappa = {fleiss_kappa(table):.3f}; all three SAILs identical for {same}/{len(by_sid)}")

# 5. Cost per correct assessment ------------------------------------------------------------------------
print("\n5. Claude API cost per correct SAIL (US$, batch prices; both languages)")
for m in MODELS[:4]:
    parts = []
    for c in CONDS:
        g = groups[(m, c, "en")] + groups[(m, c, "tr")]
        cost = sum(float(r["cost_usd"]) for r in g)
        ok = sum(r["ok_sail"] == "1" for r in g)
        parts.append(f"{LABEL[c]} {cost / ok:.4f}")
    print(f"{m:22s} " + "; ".join(parts))
print(f"total cost of the main Claude runs: US${sum(float(r['cost_usd']) for r in A):.2f}")
