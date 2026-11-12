import csv
D = "minimal_dataset"
CLS = "SAIL scale I < II < III < IV < V < VI < certified (certified = outside the SORA specific category)"
common = {
 "key": ("text", "Unique assessment id: sid|lang|condition|model|rep"),
 "sid": ("text", "Scenario id S00000-S01499; links to scenarios.csv and scenario_descriptions.jsonl"),
 "rep": ("integer", "Repeat number 0-2 (repeatability run only)"),
 "model": ("text", "Model under test: claude-haiku-4-5, claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5 (Anthropic API); gpt-oss:20b, mistral-small3.2:24b, gemma4:31b, qwen3.8:27b (Ollama tags, 4-bit builds)"),
 "lang": ("text", "Language of the description and prompt: en = English, tr = Turkish"),
}
A = {
 "model_type": ("text", "Claude API or open-weight"),
 "condition": ("text", "Set-up: closed_book = task rules only; reference = rule card added to the prompt; extract = model fills a parameter schema and the rule engine computes the classes (extract + calculator)"),
 "parsed": ("0/1", "1 = a JSON answer (or parameter set) could be read from the response"),
 "api_error": ("0/1", "1 = the API call failed (none occurred)"),
 "truncated": ("0/1", "1 = output stopped at the token limit (stop_reason max_tokens or length; none occurred)"),
 "calc_error": ("0/1", "extract only: 1 = the rule engine rejected the extracted parameters, so no classes were computed"),
 "sail_under": ("0/1", "PRIMARY OUTCOME. 1 = predicted SAIL lower than the true SAIL on the " + CLS + "; unreadable answers count as 0"),
 "sail_over": ("0/1", "1 = predicted SAIL higher than the true SAIL"),
 "sail_missing": ("0/1", "1 = no SAIL could be read (unparsed answer or rejected extraction)"),
 "sail_steps_under": ("integer", "Number of SAIL steps below the true SAIL (0 if not under-classified)"),
 "sail_err_source": ("text", "Source of a wrong SAIL (blank if SAIL correct): ground = final GRC wrong, residual ARC right; air = residual ARC wrong, final GRC right; ground+air = both wrong; lookup = both right but Table 7 look-up wrong; unparsed = no SAIL. Final GRCs of 2 or less count as equal (Table 7 treats them alike)"),
 "grc_under": ("0/1", "1 = predicted final GRC lower than the true final GRC"),
 "arc_under": ("0/1", "1 = predicted residual ARC lower than the true residual ARC"),
 "missed_certified": ("0/1", "1 = true SAIL is certified but the model gave SAIL I-VI"),
 "false_certified": ("0/1", "1 = true SAIL is I-VI but the model answered certified"),
 "self_consistent": ("0/1/blank", "1 = the model's SAIL equals the Table 7 look-up of its own final GRC and residual ARC; blank if either was missing"),
 "mentions_sora20": ("0/1", "1 = response text mentions SORA 2.0 (regex 'SORA (v/edition/sürüm) 2.0')"),
 "mentions_kinetic": ("0/1", "1 = response text mentions kinetic energy (a SORA 2.0 concept; EN or TR)"),
 "mentions_m3": ("0/1", "1 = response text mentions M3 (the SORA 2.0 emergency response plan mitigation, removed in 2.5)"),
 "ex_parsed": ("0/1/blank", "extract only: 1 = a parameter set was returned; blank for other set-ups"),
 "ex_mtom": ("0/1/blank", "extract only: maximum take-off mass correct (within 2%)"),
 "ex_dim": ("0/1/blank", "extract only: maximum characteristic dimension correct (within 2%)"),
 "ex_speed": ("0/1/blank", "extract only: maximum speed correct (within 2%)"),
 "ex_band": ("0/1/blank", "extract only: population density band correct (also 1 when the band label is right but controlled_ground_area is wrongly true)"),
 "ex_mitigations": ("0/1/blank", "extract only: integrity and assurance of M1(A), M1(B), M1(C) and M2 all correct"),
 "ex_height_side": ("0/1/blank", "extract only: top of the operational volume on the correct side of 500 ft (152.4 m)"),
 "ex_airspace": ("0/1/blank", "extract only: all six airspace flags correct (strict version; not reported in the paper)"),
 "ex_airspace_relevant": ("0/1/blank", "extract only: every airspace fact on the true Figure 6 decision path correct (version reported in the paper)"),
 "ex_flightmode": ("0/1/blank", "extract only: VLOS and airspace observer flags correct"),
 "input_tokens": ("integer", "Input tokens incl. cached prompt tokens"),
 "output_tokens": ("integer", "Output tokens incl. thinking"),
 "thinking_tokens": ("integer/blank", "Claude only: thinking tokens within output_tokens; blank for open-weight models"),
 "stop_reason": ("text", "end_turn / refusal (Claude), stop (Ollama)"),
 "latency_s": ("number/blank", "Open-weight only: wall-clock seconds for the request on one RTX 5090; blank for Claude (batch API)"),
 "cost_usd": ("number", "Claude only: API cost at Message Batches prices; 0 for open-weight models"),
}
for f, desc in [("igrc", "intrinsic ground risk class (1-10, or out_of_scope for a grey cell)"),
                ("final_grc", "final ground risk class after M1/M2 (blank if iGRC out_of_scope; above 7 gives SAIL certified)"),
                ("initial_arc", "initial air risk class a-d"), ("residual_arc", "residual air risk class a-d"),
                ("tmpr", "tactical mitigation performance requirement: none, low, medium, high"),
                ("sail", "SAIL: I-VI or certified")]:
    A[f"pred_{f}"] = ("text", f"Model answer: {desc}; blank if not given")
    A[f"true_{f}"] = ("text", f"Exact answer from the rule engine: {desc}")
    A[f"ok_{f}"] = ("0/1", f"1 = predicted {f} equals the exact answer")
X = {
 "params_returned": ("0/1", "1 = the model returned a parameter object"),
 "mtom_kg": ("number", "Extracted maximum take-off mass, kg"), "max_dimension_m": ("number", "Extracted maximum characteristic dimension, m"),
 "max_speed_ms": ("number", "Extracted maximum speed, m/s"), "controlled_ground_area": ("0/1", "Extracted: operation over a controlled ground area"),
 "max_population_density": ("number", "Extracted highest population density, people/km²"),
 "population_band": ("text", "Extracted Table 2 density band"),
 "atypical_airspace": ("0/1", "Extracted: atypical/segregated airspace"), "above_fl600": ("0/1", "Extracted: operation above FL600"),
 "airport_environment": ("0/1", "Extracted: airport/heliport environment"), "airport_airspace_class": ("text", "Extracted airspace class of the airport"),
 "flight_geography_top_m_agl": ("number", "Extracted top of the flight geography, m AGL"), "contingency_above_m": ("number", "Extracted height of the contingency volume above the flight geography, m"),
 "tmz_or_mode_c_veil": ("0/1", "Extracted: transponder mandatory zone or Mode-C veil"), "controlled_airspace": ("0/1", "Extracted: controlled airspace"),
 "urban": ("0/1", "Extracted: over an urban area"), "vlos": ("0/1", "Extracted: VLOS operation"),
 "airspace_observer_alongside_pilot": ("0/1", "Extracted: BVLOS with an airspace observer alongside the pilot"),
}
for k in ("m1a", "m1b", "m1c", "m2"):
    X[f"{k}_integrity"] = ("text", f"Extracted integrity level of {k.upper()} (low/medium/high; blank = not claimed)")
    X[f"{k}_assurance"] = ("text", f"Extracted assurance level of {k.upper()} (low/medium/high; blank = not claimed)")
S = {
 "special_case": ("text", "Small-drone case: micro (<=250 g and <=25 m/s, iGRC 1), micro_fast (249 g but faster than 25 m/s, trap), micro_heavy (just above 250 g, trap); blank otherwise"),
 "ua_type": ("text", "Aircraft type used in the description"), "purpose": ("text", "Purpose of the operation used in the description"),
 "density_format": ("text", "How population is given: numeric, qualitative, multiseg (several route segments), controlled (controlled ground area)"),
 "mtom_kg": ("number", "Maximum take-off mass, kg"), "dim_m": ("number", "Maximum characteristic dimension, m"),
 "speed_ms": ("number", "Maximum speed (designer maximum), m/s"), "cruise_ms": ("number/blank", "Cruise speed given as a distractor (blank if none)"),
 "densities": ("text", "Population densities in the footprint, people/km² (semicolon-separated per segment)"),
 "controlled_ground_area": ("0/1", "Operation over a controlled ground area"),
 "atypical": ("0/1", "Atypical/segregated airspace"), "above_fl600": ("0/1", "Operation above FL600"),
 "airport_env": ("0/1", "Airport/heliport environment"), "airport_class": ("text", "Airspace class of the airport (blank if none)"),
 "airport_class_bcd": ("0/1", "Airport in class B, C or D airspace"),
 "flight_geography_top_m": ("number", "Top of the flight geography, m AGL"), "contingency_extra_m": ("number", "Contingency volume above the flight geography, m"),
 "tmz_or_modec": ("0/1", "Transponder mandatory zone or Mode-C veil"), "controlled_airspace": ("0/1", "Controlled airspace"),
 "urban": ("0/1", "Over an urban area"), "vlos": ("0/1", "VLOS operation"), "airspace_observer": ("0/1", "BVLOS with an airspace observer alongside the pilot"),
 "erp_mentioned": ("0/1", "Description mentions an emergency response plan (no GRC credit in SORA 2.5)"),
 "true_column": ("integer", "Table 2 column used (0-4 = up to 1, 3, 8, 20, 40 m / 25, 35, 75, 120, 200 m/s): left-most column matching both dimension and speed"),
 "true_dim_column": ("integer", "Column from the dimension alone"), "true_speed_column": ("integer", "Column from the speed alone"),
 "true_pop_band": ("text", "Table 2 row: controlled, <5, <50, <500, <5000, <50000, >50000 people/km² (highest in the footprint)"),
 "true_micro_250g": ("0/1", "<=250 g and <=25 m/s rule applies (iGRC 1)"),
 "true_igrc": ("text", "Exact intrinsic GRC (1-10 or out_of_scope)"),
 "m1_credit": ("integer", "Total M1(A)+M1(B)+M1(C) credit (0 to -4)"), "m2_credit": ("integer", "M2 credit (0 to -2)"),
 "grc_after_m1": ("integer", "GRC after M1 and the M1 floor"), "m1_floor": ("integer", "M1 floor: controlled-area value of the column"),
 "m1_floor_binding": ("0/1", "The floor limited the M1 reduction"),
 "true_final_grc": ("integer/blank", "Exact final GRC (M2 applied after the floor; above 7 = certified; blank if iGRC out_of_scope)"),
 "true_initial_arc": ("text", "Exact initial ARC (Figure 6, using the top of the operational volume)"),
 "aec": ("integer", "Airspace encounter category 1-12 (Figure 6)"), "ops_top_m": ("number", "Top of the operational volume (flight geography + contingency), m AGL"),
 "above_500ft": ("0/1", "Operational volume reaches above 500 ft (152.4 m)"),
 "vlos_reduction_eligible": ("0/1", "VLOS or BVLOS with airspace observer"), "vlos_reduction_applied": ("0/1", "Residual ARC lowered by one class (never to ARC-a)"),
 "true_residual_arc": ("text", "Exact residual ARC"), "true_tmpr": ("text", "Exact TMPR (none under VLOS, else Table 6)"),
 "true_sail": ("text", "Exact SAIL (Table 7): I-VI or certified"),
}
for k in ("m1a", "m1b", "m1c", "m2"):
    S[f"{k}_integrity"] = ("text", f"Claimed integrity level of {k.upper()} (blank = not claimed)")
    S[f"{k}_assurance"] = ("text", f"Claimed assurance level of {k.upper()} (blank = not claimed)")
    S[f"robustness_{k}"] = ("text", f"Robustness of {k.upper()} = lower of integrity and assurance (none = not claimed)")
FL = {
 "speed_drives_column": "Maximum speed puts the aircraft in a higher Table 2 column than its dimension",
 "micro_250g": "<=250 g and <=25 m/s rule applies (iGRC 1)",
 "micro_trap": "Looks like a 250 g drone but the rule does not apply (too fast or just over 250 g)",
 "multiseg": "Route crosses several population segments; the highest density counts",
 "qualitative": "Population given only as a qualitative description",
 "controlled_area": "Controlled ground area",
 "cruise_distractor": "A cruise speed below the maximum speed is also given",
 "ia_mismatch": "Integrity and assurance of a mitigation differ (robustness = the lower)",
 "na_credit": "A claimed mitigation sits at a robustness level that earns no credit (N/A in Table 5)",
 "m1_floor": "The M1 floor limits the GRC reduction",
 "contingency_trap": "Flight geography below 500 ft but the contingency volume takes the operation above 500 ft",
 "vlos_reduction": "VLOS (or airspace observer) lowers the residual ARC by one class",
 "vlos_no_arc_a": "VLOS reduction would apply but the initial ARC is b, which cannot become ARC-a",
 "airport_precedence": "Airport environment and above 500 ft (airport branch of Figure 6 takes precedence)",
 "airport_class_e_g": "Airport in class E or G airspace (ARC-c, not ARC-d)",
 "erp_distractor": "An emergency response plan is mentioned (SORA 2.0 M3; no credit in 2.5)",
 "out_of_scope": "True SAIL is certified (outside the specific category)",
 "m2_below_floor": "M2 applied after the floor takes the final GRC below the M1 floor",
 "any_mitigation": "At least one ground-risk mitigation is claimed",
}
for k, v in FL.items():
    S[f"flag_{k}"] = ("0/1", "Trap flag: " + v)
J = {"sid": common["sid"], "text_en": ("text", "English operation description given to the model"),
     "text_tr": ("text", "Turkish description with the same facts")}

rows = []
for fname, extra in [("assessments.csv", A), ("repeatability_haiku55.csv", A), ("extractions.csv", X), ("scenarios.csv", S)]:
    cols = next(csv.reader(open(f"{D}/{fname}", encoding="utf-8")))
    for c in cols:
        t, d = common.get(c) or extra[c]          # KeyError = undocumented column
        rows.append({"file": fname, "column": c, "type": t, "description": d})
for c, (t, d) in J.items():
    rows.append({"file": "scenario_descriptions.jsonl", "column": c, "type": t, "description": d})
with open(f"{D}/data_dictionary.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "column", "type", "description"]); w.writeheader(); w.writerows(rows)
print(len(rows), "dictionary rows")
