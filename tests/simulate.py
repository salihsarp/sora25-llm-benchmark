"""Offline dry run of the full pipeline with two simulated 'models' (no API):
  oracle  - perfect answers;
  naive   - a rule-follower with known mistakes: picks the column by dimension only, ignores the M1
            floor, gives credit at N/A robustness, ignores the contingency volume and the VLOS reduction.
The analysis must report 100% for the oracle and attribute the naive model's errors to the right traps.
  python tests/simulate.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soraeval.generator import generate, save_jsonl
from soraeval.sora25 import (IGRC_TABLE, MITIGATION_CREDIT, LEVELS, sail_lookup, DIM_COLS, operation_from_dict,
                             step4_initial_arc)
from soraeval.runner import make_record
from soraeval import analyze

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sim")


def naive_answer(sc):
    o, t = sc["op"], sc["truth"]
    col = next(i for i, d in enumerate(DIM_COLS) if o["dim_m"] <= d)          # mistake 1: dimension only
    band = t["pop_band"]
    igrc = IGRC_TABLE[band][col] if not t["micro_250g"] else 1
    if igrc is None:
        return {"igrc": "out_of_scope", "final_grc": None, "initial_arc": t["initial_arc"],
                "residual_arc": t["initial_arc"], "tmpr": "none", "sail": "certified"}
    credit = 0
    for k in ("m1a", "m1b", "m1c", "m2"):
        m = o[k]
        if m:
            r = LEVELS[max(LEVELS.index(m["integrity"]), LEVELS.index(m["assurance"]))]  # mistake 2: max not min
            c = MITIGATION_CREDIT[k][r]
            credit += c if c is not None else -1                                  # mistake 3: credit at N/A
    final = max(1, igrc + credit)                                                 # mistake 4: no M1 floor
    op = operation_from_dict({**o, "contingency_extra_m": 0.0})                   # mistake 5: ignore contingency
    arc = step4_initial_arc(op)["initial_arc"]
    sail = "certified" if final > 7 else sail_lookup(final, arc)                  # mistake 6: no VLOS reduction
    return {"igrc": igrc, "final_grc": final, "initial_arc": arc, "residual_arc": arc,
            "tmpr": "none" if o["vlos"] else {"a": "none", "b": "low", "c": "medium", "d": "high"}[arc], "sail": sail}


def main():
    os.makedirs(OUT, exist_ok=True)
    scs = generate(300, seed=11)
    scen = os.path.join(OUT, "scenarios.jsonl")
    save_jsonl(scs, scen)
    res = os.path.join(OUT, "results.jsonl")
    with open(res, "w", encoding="utf-8") as fh:
        for sc in scs:
            for lang in ("en", "tr"):
                t = sc["truth"]
                perfect = {k: t[k] for k in ("igrc", "final_grc", "initial_arc", "residual_arc", "tmpr", "sail")}
                for model, ans in (("oracle", perfect), ("naive", naive_answer(sc))):
                    text = "reasoning\n```json\n" + json.dumps(ans) + "\n```"
                    resp = {"text": text, "usage": {}, "stop_reason": "end_turn", "latency_s": 0, "error": None}
                    fh.write(json.dumps(make_record(sc, lang, "reference", model, 0, resp), ensure_ascii=False) + "\n")
    md_path = os.path.join(OUT, "summary.md")
    analyze.main(["--scenarios", scen, "--results", res, "--out", md_path])


if __name__ == "__main__":
    main()
