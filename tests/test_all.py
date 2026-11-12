"""Offline tests (no API calls). Run:  python tests/test_all.py   (or pytest)."""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soraeval.sora25 import (Operation, Mitigation, InvalidOperation, assess, sail_lookup, SAIL_TABLE,
                             IGRC_TABLE, operation_from_dict)
from soraeval.generator import generate, render, _fmt
from soraeval.parse import extract_json, normalise_answer, calculator_answer
from soraeval import prompts

M = Mitigation


def op(**kw):
    base = dict(mtom_kg=5.0, dim_m=2.5, speed_ms=30.0, densities=[1200], flight_geography_top_m=100)
    base.update(kw)
    return Operation(**base)


# ---------------------------------------------------------------- engine: Step 2
def test_basic_case():
    t = assess(op())
    assert t["column"] == 1 and t["pop_band"] == "<5000" and t["igrc"] == 6
    assert t["final_grc"] == 6 and t["initial_arc"] == "b" and t["residual_arc"] == "b"
    assert t["sail"] == "V" and t["tmpr"] == "low"


def test_speed_drives_column():
    t = assess(op(dim_m=0.8, speed_ms=50, densities=[30]))
    assert t["dim_column"] == 0 and t["speed_column"] == 2 and t["column"] == 2 and t["igrc"] == 5


def test_250g_rule_and_traps():
    assert assess(op(mtom_kg=0.249, dim_m=0.25, speed_ms=20, densities=[20000]))["igrc"] == 1
    assert assess(op(mtom_kg=0.249, dim_m=0.25, speed_ms=30, densities=[20000]))["igrc"] == 7   # too fast
    assert assess(op(mtom_kg=0.29, dim_m=0.25, speed_ms=20, densities=[20000]))["igrc"] == 6    # too heavy


def test_highest_density_used():
    assert assess(op(densities=[30, 3000, 300]))["pop_band"] == "<5000"


def test_table2_values():
    for i, row in enumerate(["<5", "<50", "<500", "<5000", "<50000"]):
        assert IGRC_TABLE[row] == tuple(range(2 + i, 7 + i))
    assert IGRC_TABLE["controlled"] == (1, 1, 2, 3, 3)
    assert IGRC_TABLE[">50000"] == (7, 8, None, None, None)


def test_grey_cell_out_of_scope():
    t = assess(op(dim_m=5, speed_ms=50, densities=[80000]))
    assert t["igrc"] == "out_of_scope" and t["sail"] == "certified"


# ---------------------------------------------------------------- engine: Step 3
def test_controlled_area_floor():
    t = assess(op(controlled_ground_area=True, densities=[], m1a=M("low", "low")))
    assert t["igrc"] == 1 and t["final_grc"] == 1 and t["m1_floor_binding"]


def test_m1_floor_then_m2():
    t = assess(op(mtom_kg=900, dim_m=15, speed_ms=100, densities=[3],
                  m1a=M("low", "low"), m1b=M("high", "high"), m2=M("high", "high")))
    assert t["igrc"] == 5 and t["grc_after_m1"] == 3 and t["m1_floor_binding"] and t["final_grc"] == 1


def test_robustness_is_minimum_and_na():
    t = assess(op(m2=M("high", "low")))          # -> low -> N/A for M2
    assert t["robustness"]["m2"] == "low" and t["m2_credit"] == 0 and t["final_grc"] == 6
    t = assess(op(m1b=M("low", "low")))          # N/A
    assert t["final_grc"] == 6
    t = assess(op(m1a=M("medium", "high"), m2=M("high", "medium")))
    assert t["final_grc"] == 6 - 2 - 1


def test_invalid_combination():
    try:
        assess(op(m1a=M("medium", "medium"), m1b=M("medium", "medium")))
        assert False
    except InvalidOperation:
        pass


def test_final_grc_above_7():
    t = assess(op(mtom_kg=900, dim_m=15, speed_ms=100, densities=[3000]))       # iGRC 8
    assert t["igrc"] == 8 and t["sail"] == "certified"
    t = assess(op(mtom_kg=900, dim_m=15, speed_ms=100, densities=[3000], m1b=M("medium", "medium")))
    assert t["final_grc"] == 7 and t["sail"] == "VI"


# ---------------------------------------------------------------- engine: Steps 4-6
def test_arc_tree():
    assert assess(op(atypical=True))["initial_arc"] == "a"
    assert assess(op(above_fl600=True))["initial_arc"] == "b"
    assert assess(op(airport_env=True, airport_class_bcd=True))["initial_arc"] == "d"
    assert assess(op(airport_env=True, airport_class_bcd=False, flight_geography_top_m=400))["initial_arc"] == "c"
    hi = dict(flight_geography_top_m=300)
    assert assess(op(tmz_or_modec=True, **hi))["initial_arc"] == "d"
    assert assess(op(controlled_airspace=True, **hi))["initial_arc"] == "d"
    assert assess(op(urban=True, **hi))["initial_arc"] == "c"
    assert assess(op(**hi))["initial_arc"] == "c"
    assert assess(op(tmz_or_modec=True))["initial_arc"] == "c"
    assert assess(op(controlled_airspace=True))["initial_arc"] == "c"
    assert assess(op(urban=True))["initial_arc"] == "c"
    assert assess(op())["initial_arc"] == "b"


def test_contingency_counts_for_500ft():
    assert assess(op(flight_geography_top_m=130, contingency_extra_m=40))["initial_arc"] == "c"
    assert assess(op(flight_geography_top_m=130, contingency_extra_m=0))["initial_arc"] == "b"


def test_vlos_reduction():
    t = assess(op(airport_env=True, airport_class_bcd=True, vlos=True))
    assert t["initial_arc"] == "d" and t["residual_arc"] == "c" and t["tmpr"] == "none"
    t = assess(op(vlos=True))
    assert t["initial_arc"] == "b" and t["residual_arc"] == "b"            # never to ARC-a
    t = assess(op(urban=True, airspace_observer=True))
    assert t["residual_arc"] == "b" and t["tmpr"] == "low"                  # BVLOS still needs TMPR
    t = assess(op(urban=True))
    assert t["residual_arc"] == "c" and t["tmpr"] == "medium"


def test_sail_table():
    expect = {2: "I II IV VI", 3: "II II IV VI", 4: "III III IV VI", 5: "IV IV IV VI", 6: "V V V VI", 7: "VI VI VI VI"}
    for g, row in expect.items():
        assert " ".join(SAIL_TABLE[g]) == row
        for j, arc in enumerate("abcd"):
            assert sail_lookup(g, arc) == row.split()[j]
    assert sail_lookup(1, "a") == "I" and sail_lookup(8, "a") == "certified"


# ---------------------------------------------------------------- generator
SCS = generate(400, seed=7)


def test_generator_truth_reproducible():
    for sc in SCS:
        t = assess(operation_from_dict(sc["op"]))
        assert t["sail"] == sc["truth"]["sail"] and t["igrc"] == sc["truth"]["igrc"]


def test_generator_no_boundary_values():
    for sc in SCS:
        o = sc["op"]
        assert all(abs(o["dim_m"] - d) > 0.04 for d in (1, 3, 8, 20, 40))
        assert all(abs(o["speed_ms"] - s) > 0.9 for s in (25, 35, 75, 120, 200))
        top = o["flight_geography_top_m"] + o["contingency_extra_m"]
        assert o["above_fl600"] or not (135 < top < 170)
        for d in o["densities"]:
            assert all(abs(d - b) / b > 0.05 for b in (5, 50, 500, 5000, 50000))


def test_generator_coverage():
    flags = {}
    for sc in SCS:
        for k, v in sc["flags"].items():
            flags[k] = flags.get(k, 0) + int(v)
    for k in ("speed_drives_column", "multiseg", "qualitative", "na_credit", "ia_mismatch", "m1_floor",
              "contingency_trap", "vlos_reduction", "vlos_no_arc_a", "out_of_scope", "erp_distractor"):
        assert flags[k] >= 5, (k, flags[k])
    sails = {sc["truth"]["sail"] for sc in SCS}
    assert {"II", "III", "IV", "V", "VI", "certified"} <= sails


def test_render_parallel_facts():
    for sc in SCS[:80]:
        en, tr = sc["text_en"], sc["text_tr"]
        o = sc["op"]
        assert _fmt(o["speed_ms"], "en") in en and _fmt(o["speed_ms"], "tr") in tr
        assert _fmt(o["dim_m"], "en", 2) in en and _fmt(o["dim_m"], "tr", 2) in tr
        for k in ("m1a", "m1b", "m1c", "m2"):
            tag = {"m1a": "M1(A)", "m1b": "M1(B)", "m1c": "M1(C)", "m2": "M2 "}[k]
            assert (tag in en) == bool(o[k]) and (tag in tr) == bool(o[k])
        assert ("VLOS" in en) and ("VLOS" in tr)


def test_number_format():
    assert _fmt(1234.5, "en") == "1,234.5" and _fmt(1234.5, "tr") == "1.234,5"
    assert _fmt(12000, "tr", 0) == "12.000" and _fmt(2.0, "en", 2) == "2"


# ---------------------------------------------------------------- parsing / calculator
def oracle_params(sc):
    o = sc["op"]

    def mit(m):
        return None if not m else {"integrity": m["integrity"], "assurance": m["assurance"]}
    dens = None if sc["dens_fmt"] in ("qualitative", "controlled") else max(o["densities"])
    return {"mtom_kg": o["mtom_kg"], "max_dimension_m": o["dim_m"], "max_speed_ms": o["speed_ms"],
            "controlled_ground_area": o["controlled_ground_area"], "max_population_density": dens,
            "population_band": sc["truth"]["pop_band"],
            "mitigations": {k: mit(o[k]) for k in ("m1a", "m1b", "m1c", "m2")},
            "atypical_airspace": o["atypical"], "above_fl600": o["above_fl600"],
            "airport_environment": o["airport_env"], "airport_airspace_class": sc["airport_cls"],
            "flight_geography_top_m_agl": o["flight_geography_top_m"], "contingency_above_m": o["contingency_extra_m"],
            "tmz_or_mode_c_veil": o["tmz_or_modec"], "controlled_airspace": o["controlled_airspace"],
            "urban": o["urban"], "vlos": o["vlos"], "airspace_observer_alongside_pilot": o["airspace_observer"]}


def test_calculator_oracle_extraction_is_exact():
    for sc in SCS:
        a = calculator_answer(oracle_params(sc))
        t = sc["truth"]
        assert a["sail"] == t["sail"], (sc["id"], a, t["sail"])
        assert a["residual_arc"] == t["residual_arc"] and a["tmpr"] == t["tmpr"]
        assert a["igrc"] == t["igrc"]


def test_extract_json_variants():
    txt = 'reasoning...\n```json\n{"igrc": 6, "final_grc": 5, "initial_arc": "ARC-c", "residual_arc": "b", "tmpr": "Low", "sail": "SAIL IV"}\n```'
    a = normalise_answer(extract_json(txt))
    assert a == {"parsed": True, "igrc": 6, "final_grc": 5, "initial_arc": "c", "residual_arc": "b",
                 "tmpr": "low", "sail": "IV"}
    a = normalise_answer(extract_json('final: {"igrc": "out of scope", "final_grc": null, "initial_arc": "d", '
                                      '"residual_arc": "d", "tmpr": "none", "sail": "Certified category"}'))
    assert a["igrc"] == "out_of_scope" and a["sail"] == "certified" and a["final_grc"] is None
    assert normalise_answer(extract_json("no json here")) == {"parsed": False}
    j = extract_json('```json\n{"mtom_kg": 5, // kg\n "vlos": true}\n```')
    assert j == {"mtom_kg": 5, "vlos": True}


def test_prompts_build():
    for cond in prompts.CONDITIONS:
        for lang in ("en", "tr"):
            s, u = prompts.build(cond, lang, "x")
            assert isinstance(s, list) and u.endswith("x")
    assert len(prompts.build("reference", "en", "x")[0]) == 2


def test_reference_card_matches_engine():
    """Every iGRC and SAIL number printed in the reference cards must equal the engine tables."""
    import re
    for card in (prompts.REFERENCE_EN, prompts.REFERENCE_TR):
        rows = re.findall(r"(?m)^\s+(?:<\s*[\d.,]+|Controlled ground area|Kontrollü yer alanı)[^:]*:\s+([\dx ]+)$", card)
        nums = [r.split() for r in rows]
        exp = [IGRC_TABLE[b] for b in ("controlled", "<5", "<50", "<500", "<5000", "<50000")]
        assert [tuple(int(v) for v in r) for r in nums[:6]] == exp, nums
        sail_rows = re.findall(r"(?m)^\s+(<=2|[3-7]):\s+(.+)$", card)
        for g, vals in sail_rows:
            gi = 2 if g == "<=2" else int(g)
            assert tuple(vals.split()) == SAIL_TABLE[gi]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
