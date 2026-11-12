"""Grade results against the SORA 2.5 ground truth and write a Markdown summary (standard library only).

  python -m soraeval.analyze --results results/pilot.jsonl [results/other.jsonl ...] --out results/summary_pilot.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

from .generator import load_jsonl
from .sora25 import SAIL_ORDER, ARC_ORDER, sail_lookup

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ("igrc", "final_grc", "initial_arc", "residual_arc", "tmpr", "sail")
# which step each trap is expected to break
TRAP_STEP = {
    "speed_drives_column": "igrc", "micro_250g": "igrc", "micro_trap": "igrc", "multiseg": "igrc",
    "qualitative": "igrc", "controlled_area": "final_grc", "cruise_distractor": "igrc",
    "ia_mismatch": "final_grc", "na_credit": "final_grc", "m1_floor": "final_grc", "erp_distractor": "final_grc",
    "contingency_trap": "initial_arc", "airport_precedence": "initial_arc", "airport_class_e_g": "initial_arc",
    "vlos_reduction": "residual_arc", "vlos_no_arc_a": "residual_arc", "out_of_scope": "sail",
    "m2_below_floor": "final_grc",
}


# ------------------------------------------------------------------ statistics (no dependencies)
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def binom_two_sided(k, n):
    """Exact two-sided binomial test p-value for p = 0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar(b, c):
    """Exact McNemar on discordant pairs b (A right, B wrong) and c (A wrong, B right)."""
    return binom_two_sided(b, b + c)


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def pct(k, n):
    if n == 0:
        return "-"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] (n={n})"


# ------------------------------------------------------------------ grading
def grade(rec, sc):
    t, a = sc["truth"], rec.get("answer") or {}
    g = {"key": rec["key"], "sid": rec["sid"], "lang": rec["lang"], "condition": rec["condition"],
         "model": rec["model"], "rep": rec.get("rep", 0), "parsed": bool(a.get("parsed")),
         "api_error": bool(rec.get("error")), "cost_usd": rec.get("cost_usd", 0.0),
         "calc_error": bool(a.get("calc_error")), "truncated": rec.get("stop_reason") in ("max_tokens", "length")}
    for f in FIELDS:
        pv, tv = a.get(f), t[f]
        if f == "final_grc" and t["sail"] == "certified" and t["igrc"] == "out_of_scope":
            g[f"ok_{f}"] = pv in (None, "out_of_scope")
        elif f == "final_grc" and isinstance(tv, int) and tv <= 2 and isinstance(pv, int):
            g[f"ok_{f}"] = pv == tv          # exact; SAIL treats <=2 alike, reported separately
        else:
            g[f"ok_{f}"] = pv == tv
        g[f"pred_{f}"] = pv
        g[f"true_{f}"] = tv
    ps = a.get("sail")
    to = SAIL_ORDER[t["sail"]]
    po = SAIL_ORDER.get(ps) if ps else None
    g["sail_under"] = po is not None and po < to
    g["sail_over"] = po is not None and po > to
    g["sail_missing"] = po is None
    g["sail_steps_under"] = (to - po) if (po is not None and po < to) else 0
    # GRC / ARC direction
    pf, tf = a.get("final_grc"), t["final_grc"]
    g["grc_under"] = isinstance(pf, int) and isinstance(tf, int) and pf < tf
    pr, tr = a.get("residual_arc"), t["residual_arc"]
    g["arc_under"] = pr in ARC_ORDER and ARC_ORDER[pr] < ARC_ORDER[tr]
    # in-scope operation reported as SAIL (missed 'certified') and the reverse
    g["missed_certified"] = t["sail"] == "certified" and ps not in (None, "certified")
    g["false_certified"] = t["sail"] != "certified" and ps == "certified"
    # internal consistency of the model's own SAIL with its own GRC/ARC
    if isinstance(pf, int) and pr in ARC_ORDER and ps:
        g["self_consistent"] = sail_lookup(pf, pr) == ps
    else:
        g["self_consistent"] = None
    # SAIL error source
    if not g["ok_sail"]:
        ground_ok = g["ok_final_grc"] or (isinstance(pf, int) and isinstance(tf, int) and pf <= 2 and tf <= 2)
        air_ok = g["ok_residual_arc"]
        g["sail_err_source"] = ("unparsed" if po is None else "ground+air" if not ground_ok and not air_ok else
                                "ground" if not ground_ok else "air" if not air_ok else "lookup")
    else:
        g["sail_err_source"] = ""
    for k, v in sc["flags"].items():
        g[f"flag_{k}"] = bool(v)
    g["true_sail"] = t["sail"]
    # version-confusion markers in free text (closed_book / reference)
    raw = rec.get("raw") or ""
    g["mentions_m3"] = bool(re.search(r"\bM3\b", raw))
    g["mentions_sora20"] = bool(re.search(r"(?i)SORA\s*(v|edition|sürüm)?\s*2\.0", raw))
    g["mentions_kinetic"] = bool(re.search(r"(?i)kinetic energy|kinetik enerji", raw))
    # extraction accuracy
    if rec["condition"] == "extract":
        g.update(extraction_checks(rec.get("params"), sc))
    return g


def _eqf(a, b, tol=0.02):
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return False


def _airspace_path_ok(p, o):
    """True when every airspace fact that the Figure 6 tree reads on the scenario's TRUE path was extracted correctly
    (atypical -> above FL600 -> airport environment [-> airport class B-D] -> TMZ/Mode-C veil -> controlled -> urban).
    Facts the true path never reaches are ignored; the 500 ft height side is checked separately (ex_height_side)."""
    cls = p.get("airport_airspace_class")
    pred = {"atypical": bool(p.get("atypical_airspace")), "above_fl600": bool(p.get("above_fl600")),
            "airport_env": bool(p.get("airport_environment")),
            "airport_class_bcd": (str(cls).strip().upper() if cls else None) in ("A", "B", "C", "D"),
            "tmz_or_modec": bool(p.get("tmz_or_mode_c_veil")), "controlled_airspace": bool(p.get("controlled_airspace")),
            "urban": bool(p.get("urban"))}
    for k in ("atypical", "above_fl600", "airport_env"):
        if pred[k] != bool(o[k]):
            return False
        if o[k]:
            return k != "airport_env" or pred["airport_class_bcd"] == bool(o["airport_class_bcd"])
    for k in ("tmz_or_modec", "controlled_airspace", "urban"):
        if pred[k] != bool(o[k]):
            return False
        if o[k]:
            return True
    return True


def extraction_checks(p, sc):
    o, t = sc["op"], sc["truth"]
    if not isinstance(p, dict):
        return {"ex_parsed": False}
    mits = p.get("mitigations") or {}

    def mit_ok(k):
        tv, pv = o[k], mits.get(k)
        if not tv:
            return pv in (None, False, "none", {})
        if isinstance(pv, dict):
            return (str(pv.get("integrity")).lower() == tv["integrity"] and
                    str(pv.get("assurance")).lower() == tv["assurance"])
        return False
    band = p.get("population_band")
    band = str(band).replace(",", "").replace(".", "").replace(" ", "") if band is not None else None
    top = (p.get("flight_geography_top_m_agl") or 0) + (p.get("contingency_above_m") or 0)
    return {
        "ex_parsed": True,
        "ex_mtom": _eqf(p.get("mtom_kg"), o["mtom_kg"]),
        "ex_dim": _eqf(p.get("max_dimension_m"), o["dim_m"]),
        "ex_speed": _eqf(p.get("max_speed_ms"), o["speed_ms"]),
        "ex_band": (band == t["pop_band"]) or (p.get("controlled_ground_area") is True and t["pop_band"] == "controlled"),
        "ex_mitigations": all(mit_ok(k) for k in ("m1a", "m1b", "m1c", "m2")),
        "ex_height_side": o["above_fl600"] or ((top > 152.4) == t["above_500ft"]),
        "ex_airspace": (bool(p.get("atypical_airspace")) == o["atypical"] and bool(p.get("above_fl600")) == o["above_fl600"]
                        and bool(p.get("airport_environment")) == o["airport_env"]
                        and bool(p.get("tmz_or_mode_c_veil")) == o["tmz_or_modec"]
                        and bool(p.get("controlled_airspace")) == o["controlled_airspace"]
                        and bool(p.get("urban")) == o["urban"]),
        "ex_airspace_relevant": _airspace_path_ok(p, o),
        "ex_flightmode": (bool(p.get("vlos")) == o["vlos"] and
                          bool(p.get("airspace_observer_alongside_pilot")) == o["airspace_observer"]),
    }


# ------------------------------------------------------------------ report
def _rows(G, keyf):
    out = defaultdict(list)
    for g in G:
        out[keyf(g)].append(g)
    return out


def summarize(G):
    L = []
    w = L.append
    models = sorted({g["model"] for g in G})
    conds = [c for c in ("closed_book", "reference", "extract") if any(g["condition"] == c for g in G)]
    langs = [l for l in ("en", "tr") if any(g["lang"] == l for g in G)]
    w(f"# SORA 2.5 LLM benchmark - results summary\n")
    w(f"Episodes: {len(G)}; scenarios: {len({g['sid'] for g in G})}; models: {', '.join(models)}; "
      f"API errors: {sum(g['api_error'] for g in G)}; unparsed: {sum(not g['parsed'] for g in G)}; "
      f"truncated: {sum(g['truncated'] for g in G)}; cost: ${sum(g['cost_usd'] for g in G):.3f}\n")

    w("## 1. Primary outcome - SAIL under-classification (rated safer than it is), with exact SAIL accuracy\n")
    w("| model | condition | lang | SAIL exact | SAIL UNDER (primary) | SAIL over | missing | missed 'certified' |")
    w("|---|---|---|---|---|---|---|---|")
    for m in models:
        for c in conds:
            for l in langs:
                S = [g for g in G if g["model"] == m and g["condition"] == c and g["lang"] == l]
                if not S:
                    continue
                n = len(S)
                cert = [g for g in S if g["true_sail"] == "certified"]
                w(f"| {m} | {c} | {l} | {pct(sum(g['ok_sail'] for g in S), n)} | "
                  f"{pct(sum(g['sail_under'] for g in S), n)} | {pct(sum(g['sail_over'] for g in S), n)} | "
                  f"{sum(g['sail_missing'] for g in S)} | {pct(sum(g['missed_certified'] for g in cert), len(cert))} |")
    w("")
    w("Under vs over among wrong SAILs (exact binomial test of symmetry):\n")
    w("| model | condition | under | over | p (two-sided) |")
    w("|---|---|---|---|---|")
    for m in models:
        for c in conds:
            S = [g for g in G if g["model"] == m and g["condition"] == c]
            u, o = sum(g["sail_under"] for g in S), sum(g["sail_over"] for g in S)
            if S:
                w(f"| {m} | {c} | {u} | {o} | {binom_two_sided(u, u + o):.3g} |")
    w("")

    w("## 2. Step accuracy\n")
    w("| model | condition | lang | iGRC | final GRC | initial ARC | residual ARC | TMPR | SAIL | GRC under | ARC under | SAIL self-consistent |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in models:
        for c in conds:
            for l in langs:
                S = [g for g in G if g["model"] == m and g["condition"] == c and g["lang"] == l]
                if not S:
                    continue
                n = len(S)
                cells = [f"{100 * sum(g['ok_' + f] for g in S) / n:.1f}" for f in FIELDS]
                sc_ = [g for g in S if g["self_consistent"] is not None]
                w(f"| {m} | {c} | {l} | " + " | ".join(cells) +
                  f" | {100 * sum(g['grc_under'] for g in S) / n:.1f} | {100 * sum(g['arc_under'] for g in S) / n:.1f} | "
                  f"{(100 * sum(g['self_consistent'] for g in sc_) / len(sc_)) if sc_ else float('nan'):.1f} |")
    w("")

    w("## 3. Where SAIL errors come from\n")
    w("| model | condition | SAIL errors | ground only | air only | ground+air | lookup (inputs right) | unparsed |")
    w("|---|---|---|---|---|---|---|---|")
    for m in models:
        for c in conds:
            S = [g for g in G if g["model"] == m and g["condition"] == c and not g["ok_sail"]]
            if not S:
                continue
            cnt = defaultdict(int)
            for g in S:
                cnt[g["sail_err_source"]] += 1
            w(f"| {m} | {c} | {len(S)} | {cnt['ground']} | {cnt['air']} | {cnt['ground+air']} | {cnt['lookup']} | {cnt['unparsed']} |")
    w("")

    w("## 4. Traps - error rate of the targeted step, flagged vs not flagged (pooled over languages)\n")
    for m in models:
        for c in conds:
            S = [g for g in G if g["model"] == m and g["condition"] == c]
            if not S:
                continue
            w(f"**{m} / {c}**\n")
            w("| trap | step | error if trap | error if no trap | difference (pp) |")
            w("|---|---|---|---|---|")
            for trap, step in TRAP_STEP.items():
                A = [g for g in S if g.get(f"flag_{trap}")]
                B = [g for g in S if not g.get(f"flag_{trap}")]
                if len(A) < 3:
                    continue
                ea = sum(not g[f"ok_{step}"] for g in A) / len(A)
                eb = sum(not g[f"ok_{step}"] for g in B) / max(1, len(B))
                w(f"| {trap} | {step} | {100 * ea:.1f}% (n={len(A)}) | {100 * eb:.1f}% (n={len(B)}) | {100 * (ea - eb):+.1f} |")
            w("")

    w("## 5. Paired comparisons (exact McNemar on SAIL exact; Holm within each family)\n")
    idx = {(g["model"], g["condition"], g["lang"], g["sid"], g["rep"]): g for g in G}
    fam = []
    for m in models:
        for l in langs:
            for c1, c2 in (("closed_book", "reference"), ("reference", "extract"), ("closed_book", "extract")):
                if c1 not in conds or c2 not in conds:
                    continue
                b = c_ = n = 0
                for (mm, cc, ll, sid, rep), g in idx.items():
                    if mm == m and cc == c1 and ll == l:
                        h = idx.get((m, c2, l, sid, rep))
                        if h is None:
                            continue
                        n += 1
                        b += int(g["ok_sail"] and not h["ok_sail"])
                        c_ += int(h["ok_sail"] and not g["ok_sail"])
                fam.append((f"{m} {l}: {c1} vs {c2}", n, b, c_, mcnemar(b, c_)))
    lang_fam = []
    if len(langs) == 2:
        for m in models:
            for c in conds:
                b = c_ = n = 0
                bu = cu = 0
                for (mm, cc, ll, sid, rep), g in idx.items():
                    if mm == m and cc == c and ll == "en":
                        h = idx.get((m, c, "tr", sid, rep))
                        if h is None:
                            continue
                        n += 1
                        b += int(g["ok_sail"] and not h["ok_sail"])
                        c_ += int(h["ok_sail"] and not g["ok_sail"])
                        bu += int(h["sail_under"] and not g["sail_under"])
                        cu += int(g["sail_under"] and not h["sail_under"])
                lang_fam.append((f"{m} {c}: EN vs TR (SAIL exact)", n, b, c_, mcnemar(b, c_)))
                lang_fam.append((f"{m} {c}: TR vs EN (SAIL under)", n, bu, cu, mcnemar(bu, cu)))
    for title, F in (("Conditions", fam), ("Language", lang_fam)):
        if not F:
            continue
        adj = holm([f[4] for f in F])
        w(f"**{title}** (b = first right & second wrong; c = reverse)\n")
        w("| comparison | pairs | b | c | p | p (Holm) |")
        w("|---|---|---|---|---|---|")
        for f, pa in zip(F, adj):
            w(f"| {f[0]} | {f[1]} | {f[2]} | {f[3]} | {f[4]:.3g} | {pa:.3g} |")
        w("")

    ex = [g for g in G if g["condition"] == "extract" and g.get("ex_parsed")]
    if ex:
        w("## 6. Extraction accuracy (extract + calculator condition)\n")
        keys = ["ex_mtom", "ex_dim", "ex_speed", "ex_band", "ex_mitigations", "ex_height_side", "ex_airspace", "ex_airspace_relevant",
                "ex_flightmode"]
        w("| model | lang | n | " + " | ".join(k[3:] for k in keys) + " | calc errors |")
        w("|---|---|---|" + "---|" * (len(keys) + 1))
        for m in models:
            for l in langs:
                S = [g for g in ex if g["model"] == m and g["lang"] == l]
                if S:
                    w(f"| {m} | {l} | {len(S)} | " + " | ".join(f"{100 * sum(g[k] for g in S) / len(S):.1f}" for k in keys)
                      + f" | {sum(g['calc_error'] for g in S)} |")
        w("")

    w("## 7. SORA 2.0 / version-confusion markers in the reasoning text\n")
    w("| model | condition | lang | mentions M3 | mentions 'SORA 2.0' | mentions kinetic energy | M3 when ERP distractor present |")
    w("|---|---|---|---|---|---|---|")
    for m in models:
        for c in ("closed_book", "reference"):
            for l in langs:
                S = [g for g in G if g["model"] == m and g["condition"] == c and g["lang"] == l]
                if not S:
                    continue
                E = [g for g in S if g["flag_erp_distractor"]]
                w(f"| {m} | {c} | {l} | {100 * sum(g['mentions_m3'] for g in S) / len(S):.1f}% | "
                  f"{100 * sum(g['mentions_sora20'] for g in S) / len(S):.1f}% | "
                  f"{100 * sum(g['mentions_kinetic'] for g in S) / len(S):.1f}% | "
                  f"{(100 * sum(g['mentions_m3'] for g in E) / len(E)) if E else float('nan'):.1f}% |")
    w("")

    w("## 8. SAIL under-classification by true SAIL\n")
    w("| model | condition | " + " | ".join(["I", "II", "III", "IV", "V", "VI", "certified"]) + " |")
    w("|---|---|" + "---|" * 7)
    for m in models:
        for c in conds:
            S = [g for g in G if g["model"] == m and g["condition"] == c]
            cells = []
            for s in ["I", "II", "III", "IV", "V", "VI", "certified"]:
                T = [g for g in S if g["true_sail"] == s]
                cells.append(f"{100 * sum(g['sail_under'] for g in T) / len(T):.0f}% ({len(T)})" if T else "-")
            if S:
                w(f"| {m} | {c} | " + " | ".join(cells) + " |")
    w("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=os.path.join(HERE, "data", "scenarios.jsonl"))
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    scs = {s["id"]: s for s in load_jsonl(args.scenarios)}
    recs = {}
    for path in args.results:
        for r in load_jsonl(path):
            if r.get("error"):
                if r["key"] not in recs:
                    recs[r["key"]] = r
                continue
            recs[r["key"]] = r          # last successful record per key wins
    G = [grade(r, scs[r["sid"]]) for r in recs.values() if r["sid"] in scs]
    md = summarize(G)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    csv_path = os.path.splitext(args.out)[0] + "_graded.csv"
    cols = sorted({k for g in G for k in g})
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for g in G:
            wr.writerow(g)
    print(md)
    print(f"\nwrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
