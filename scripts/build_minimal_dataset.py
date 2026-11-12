"""Build the minimal dataset for the SORA-LLM study from the raw run files.
Run from the repository root: python3 scripts/build_minimal_dataset.py"""
import csv, json, os, sys
sys.path.insert(0, os.getcwd())
from soraeval.analyze import grade, load_jsonl

OUT = "minimal_dataset"
os.makedirs(OUT, exist_ok=True)
MAIN = ["main_haiku45", "main_haiku55", "main_sonnet55", "main_opus55", "main_open_gptoss20b",
        "main_open_mistralsmall32", "main_open_gemma4_31b", "main_open_qwen38_27b"]
MODEL_ORDER = ["claude-haiku-4-5", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
               "gpt-oss:20b", "mistral-small3.2:24b", "gemma4:31b", "qwen3.8:27b"]
COND_ORDER = ["closed_book", "reference", "extract"]

scs = {s["id"]: s for s in load_jsonl("data/scenarios.jsonl")}


def b(v):
    """bool -> 1/0, None -> blank, others unchanged"""
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    return v


def load_runs(files):
    recs = {}
    for f in files:
        for r in load_jsonl(f"results/{f}.jsonl"):
            if r.get("error"):
                if r["key"] not in recs:
                    recs[r["key"]] = r
                continue
            recs[r["key"]] = r
    return recs


def usage(r):
    u = r.get("usage") or {}
    inp = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
    th = (u.get("output_tokens_details") or {}).get("thinking_tokens")
    return inp, u.get("output_tokens"), th


A_COLS = (["key", "sid", "model", "model_type", "condition", "lang", "parsed", "api_error", "truncated", "calc_error"]
          + [f"{p}_{f}" for f in ("igrc", "final_grc", "initial_arc", "residual_arc", "tmpr", "sail") for p in ("pred", "true", "ok")]
          + ["sail_under", "sail_over", "sail_missing", "sail_steps_under", "sail_err_source", "grc_under", "arc_under",
             "missed_certified", "false_certified", "self_consistent", "mentions_sora20", "mentions_kinetic", "mentions_m3",
             "ex_parsed", "ex_mtom", "ex_dim", "ex_speed", "ex_band", "ex_mitigations", "ex_height_side", "ex_airspace",
             "ex_airspace_relevant", "ex_flightmode", "input_tokens", "output_tokens", "thinking_tokens", "stop_reason",
             "latency_s", "cost_usd"])


def rows_for(recs, rep_col=False):
    out = []
    for r in recs.values():
        g = grade(r, scs[r["sid"]])
        inp, outp, th = usage(r)
        row = {k: b(g.get(k)) for k in A_COLS if k in g}
        row["model_type"] = "open-weight" if r["model"] in MODEL_ORDER[4:] else "Claude API"
        row["true_sail"] = g["true_sail"]
        row.update(input_tokens=inp, output_tokens=outp, thinking_tokens=b(th), stop_reason=r.get("stop_reason") or "",
                   latency_s=b(r.get("latency_s")), cost_usd=r.get("cost_usd", 0.0))
        if rep_col:
            row["rep"] = r.get("rep", 0)
        out.append((r, row))
    out.sort(key=lambda x: (MODEL_ORDER.index(x[1]["model"]), COND_ORDER.index(x[1]["condition"]), x[1]["lang"],
                            x[1]["sid"], x[0].get("rep", 0)))
    return out


def write(name, cols, rows):
    with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="raise")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})
    print(name, len(rows))


# ---------------------------------------------------------------- assessments
main = rows_for(load_runs(MAIN))
write("assessments.csv", A_COLS, [r for _, r in main])

# check against the graded CSV used for the paper
ref = {r["key"]: r for r in csv.DictReader(open("results/summary_main_graded.csv", encoding="utf-8"))}
assert len(ref) == len(main), (len(ref), len(main))
nbad = 0
for _, row in main:
    o = ref[row["key"]]
    for c in A_COLS:
        if c in o:
            ov = o[c]
            ov = {"True": 1, "False": 0}.get(ov, ov)
            nv = row.get(c, "")
            if str(ov) != str(nv) and not (c == "cost_usd" and abs(float(ov or 0) - float(nv or 0)) < 1e-9):
                nbad += 1
                if nbad < 5:
                    print("MISMATCH", row["key"], c, repr(o[c]), repr(nv))
print("mismatches vs paper graded CSV:", nbad)

# ---------------------------------------------------------------- repeatability
stab = rows_for(load_runs(["stability_haiku55"]), rep_col=True)
R_COLS = A_COLS[:2] + ["rep"] + A_COLS[2:]
write("repeatability_haiku55.csv", R_COLS, [r for _, r in stab])

# ---------------------------------------------------------------- extractions
X_COLS = ["key", "sid", "model", "lang", "params_returned", "mtom_kg", "max_dimension_m", "max_speed_ms",
          "controlled_ground_area", "max_population_density", "population_band"]
for k in ("m1a", "m1b", "m1c", "m2"):
    X_COLS += [f"{k}_integrity", f"{k}_assurance"]
X_COLS += ["atypical_airspace", "above_fl600", "airport_environment", "airport_airspace_class",
           "flight_geography_top_m_agl", "contingency_above_m", "tmz_or_mode_c_veil", "controlled_airspace", "urban",
           "vlos", "airspace_observer_alongside_pilot"]
xrows = []
for r, row in main:
    if row["condition"] != "extract":
        continue
    p = r.get("params")
    x = {"key": row["key"], "sid": row["sid"], "model": row["model"], "lang": row["lang"],
         "params_returned": int(isinstance(p, dict))}
    if isinstance(p, dict):
        for c in X_COLS[5:]:
            if c in p and not isinstance(p[c], (dict, list)):
                x[c] = b(p[c])
        mits = p.get("mitigations") or {}
        for k in ("m1a", "m1b", "m1c", "m2"):
            v = mits.get(k) if isinstance(mits, dict) else None
            if isinstance(v, dict):
                x[f"{k}_integrity"], x[f"{k}_assurance"] = v.get("integrity") or "", v.get("assurance") or ""
            elif v not in (None, False, {}, ""):
                x[f"{k}_integrity"] = json.dumps(v, ensure_ascii=False)
    xrows.append(x)
write("extractions.csv", X_COLS, xrows)

# ---------------------------------------------------------------- scenarios
used = sorted({row["sid"] for _, row in main})
LEVEL = lambda m, k: (m or {}).get(k, "") if m else ""
S_COLS = (["sid", "special_case", "ua_type", "purpose", "density_format", "mtom_kg", "dim_m", "speed_ms", "cruise_ms",
           "densities", "controlled_ground_area"]
          + [f"{k}_{q}" for k in ("m1a", "m1b", "m1c", "m2") for q in ("integrity", "assurance")]
          + ["atypical", "above_fl600", "airport_env", "airport_class", "airport_class_bcd", "flight_geography_top_m",
             "contingency_extra_m", "tmz_or_modec", "controlled_airspace", "urban", "vlos", "airspace_observer",
             "erp_mentioned",
             "true_column", "true_dim_column", "true_speed_column", "true_pop_band", "true_micro_250g", "true_igrc",
             "m1_credit", "m2_credit", "grc_after_m1", "m1_floor", "m1_floor_binding", "true_final_grc",
             "true_initial_arc", "aec", "ops_top_m", "above_500ft", "vlos_reduction_eligible", "vlos_reduction_applied",
             "true_residual_arc", "true_tmpr", "true_sail"]
          + [f"robustness_{k}" for k in ("m1a", "m1b", "m1c", "m2")])
FLAGS = list(scs[used[0]]["flags"].keys())
S_COLS += [f"flag_{k}" for k in FLAGS]
srows, drows = [], []
for sid in used:
    s = scs[sid]
    o, t = s["op"], s["truth"]
    row = {"sid": sid, "special_case": s["special"] or "", "ua_type": s["ua_type"], "purpose": s["purpose"],
           "density_format": s["dens_fmt"], "cruise_ms": b(s["cruise_ms"]),
           "densities": ";".join(str(d) for d in o["densities"]), "airport_class": s["airport_cls"] or "",
           "erp_mentioned": b(s["erp"])}
    for k in ("mtom_kg", "dim_m", "speed_ms", "controlled_ground_area", "atypical", "above_fl600", "airport_env",
              "airport_class_bcd", "flight_geography_top_m", "contingency_extra_m", "tmz_or_modec",
              "controlled_airspace", "urban", "vlos", "airspace_observer"):
        row[k] = b(o[k])
    for k in ("m1a", "m1b", "m1c", "m2"):
        row[f"{k}_integrity"], row[f"{k}_assurance"] = LEVEL(o[k], "integrity"), LEVEL(o[k], "assurance")
        row[f"robustness_{k}"] = (t.get("robustness") or {}).get(k, "")
    for k in ("column", "dim_column", "speed_column", "pop_band", "micro_250g", "igrc", "final_grc", "initial_arc",
              "residual_arc", "tmpr", "sail"):
        row[f"true_{k}"] = b(t.get(k))
    for k in ("m1_credit", "m2_credit", "grc_after_m1", "m1_floor", "m1_floor_binding", "aec", "ops_top_m",
              "above_500ft", "vlos_reduction_eligible", "vlos_reduction_applied"):
        row[k] = b(t.get(k))
    for k in FLAGS:
        row[f"flag_{k}"] = b(s["flags"][k])
    srows.append(row)
    drows.append({"sid": sid, "text_en": s["text_en"], "text_tr": s["text_tr"]})
write("scenarios.csv", S_COLS, srows)
with open(f"{OUT}/scenario_descriptions.jsonl", "w", encoding="utf-8") as fh:
    for d in drows:
        fh.write(json.dumps(d, ensure_ascii=False) + "\n")
print("scenario_descriptions.jsonl", len(drows))
print("FLAGS", FLAGS)
