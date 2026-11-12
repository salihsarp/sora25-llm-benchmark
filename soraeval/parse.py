"""Parse model outputs and normalise answers."""
from __future__ import annotations

import json
import re

from .sora25 import Operation, Mitigation, InvalidOperation, assess, pop_band, POP_BANDS

ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI",
         "I": "I", "II": "II", "III": "III", "IV": "IV", "V": "V", "VI": "VI"}


def _strip_comments(s: str) -> str:
    # remove // comments that some models copy from the schema
    return re.sub(r"(?m)//[^\n\"]*$", "", s)


def extract_json(text: str):
    """Return the last JSON object in a ```json block, else the last balanced {...} that parses."""
    if not text:
        return None
    blocks = re.findall(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, flags=re.S)
    for b in reversed(blocks):
        for cand in (b, _strip_comments(b)):
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                pass
    # fallback: scan for balanced braces from the end
    ends = [i for i, ch in enumerate(text) if ch == "}"]
    for end in reversed(ends):
        depth = 0
        for start in range(end, -1, -1):
            if text[start] == "}":
                depth += 1
            elif text[start] == "{":
                depth -= 1
                if depth == 0:
                    cand = text[start:end + 1]
                    for c in (cand, _strip_comments(cand)):
                        try:
                            return json.loads(c)
                        except json.JSONDecodeError:
                            pass
                    break
    return None


def norm_arc(v):
    if v is None:
        return None
    s = str(v).strip().lower().replace("arc", "").replace("-", "").replace("_", "").strip()
    return s if s in ("a", "b", "c", "d") else None


def norm_sail(v):
    if v is None:
        return None
    s = str(v).strip()
    low = s.lower()
    if any(k in low for k in ("cert", "category c", "out", "sertifika", ">7", "n/a")):
        return "certified"
    s = re.sub(r"(?i)sail", "", s).strip().upper()
    return ROMAN.get(s)


def norm_int(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else None
    s = str(v).strip().lower()
    if any(k in s for k in ("out", "not part", "certified", "n/a", "kapsam")):
        return "out_of_scope"
    m = re.fullmatch(r"\s*(\d+)\s*", s)
    return int(m.group(1)) if m else None


def norm_tmpr(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    for k, out in (("none", "none"), ("no ", "none"), ("not required", "none"), ("yok", "none"),
                   ("low", "low"), ("düşük", "low"), ("medium", "medium"), ("orta", "medium"),
                   ("high", "high"), ("yüksek", "high")):
        if s == k.strip() or s.startswith(k):
            return out
    if s in ("n/a", "na", "-", ""):
        return "none"
    return None


def normalise_answer(obj) -> dict:
    if not isinstance(obj, dict):
        return {"parsed": False}
    o = {str(k).lower(): v for k, v in obj.items()}
    igrc = norm_int(o.get("igrc"))
    fgrc = norm_int(o.get("final_grc"))
    return {"parsed": True, "igrc": igrc, "final_grc": fgrc if fgrc != "out_of_scope" else None,
            "initial_arc": norm_arc(o.get("initial_arc")), "residual_arc": norm_arc(o.get("residual_arc")),
            "tmpr": norm_tmpr(o.get("tmpr")), "sail": norm_sail(o.get("sail"))}


# ------------------------------------------------------------------ extract condition -> calculator
LV = ("low", "medium", "high")
LV_TR = {"düşük": "low", "orta": "medium", "yüksek": "high"}


def _lvl(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    s = LV_TR.get(s, s)
    return s if s in LV else None


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "evet", "1")
    return bool(v)


def _num(v, default=None):
    try:
        if v is None:
            return default
        if isinstance(v, str):
            v = v.replace(",", ".")
        return float(v)
    except (TypeError, ValueError):
        return default


def params_to_operation(p: dict) -> Operation:
    """Map extracted parameters to an engine Operation. Raises InvalidOperation / KeyError on bad input."""
    mits = p.get("mitigations") or {}
    m = {}
    for k in ("m1a", "m1b", "m1c", "m2"):
        v = mits.get(k)
        if v in (None, False, "none", "null", {}):
            m[k] = None
            continue
        if isinstance(v, str):
            lv = _lvl(v)
            if lv is None:
                raise InvalidOperation(f"bad mitigation level {v}")
            m[k] = Mitigation(lv, lv)
        else:
            i, a = _lvl(v.get("integrity")), _lvl(v.get("assurance"))
            if i is None and a is None:
                lv = _lvl(v.get("robustness"))
                i = a = lv
            if i is None or a is None:
                raise InvalidOperation(f"bad mitigation {v}")
            m[k] = Mitigation(i, a)
    controlled = _bool(p.get("controlled_ground_area", False))
    dens = _num(p.get("max_population_density"))
    band = p.get("population_band")
    densities = []
    if not controlled:
        if dens is not None and dens > 0:
            densities = [dens]
        else:
            band = str(band).replace(",", "").replace(".", "").replace(" ", "") if band else None
            if band == "controlled":
                controlled = True
            elif band in POP_BANDS:
                # representative density inside the band
                rep = {"<5": 2, "<50": 20, "<500": 200, "<5000": 2000, "<50000": 20000, ">50000": 80000}
                densities = [rep[band]]
            else:
                raise InvalidOperation(f"no usable population information: {band}")
    cls = p.get("airport_airspace_class")
    cls = str(cls).strip().upper() if cls else None
    fl600 = _bool(p.get("above_fl600", False))
    return Operation(
        mtom_kg=_num(p.get("mtom_kg"), 1e9), dim_m=_num(p.get("max_dimension_m"), 1e9),
        speed_ms=_num(p.get("max_speed_ms"), 1e9), densities=densities, controlled_ground_area=controlled,
        m1a=m["m1a"], m1b=m["m1b"], m1c=m["m1c"], m2=m["m2"],
        atypical=_bool(p.get("atypical_airspace", False)), above_fl600=fl600,
        airport_env=_bool(p.get("airport_environment", False)), airport_class_bcd=cls in ("A", "B", "C", "D"),
        flight_geography_top_m=_num(p.get("flight_geography_top_m_agl"), 0.0),
        contingency_extra_m=_num(p.get("contingency_above_m"), 0.0),
        tmz_or_modec=_bool(p.get("tmz_or_mode_c_veil", False)),
        controlled_airspace=_bool(p.get("controlled_airspace", False)), urban=_bool(p.get("urban", False)),
        vlos=_bool(p.get("vlos", False)), airspace_observer=_bool(p.get("airspace_observer_alongside_pilot", False)),
    )


def calculator_answer(p) -> dict:
    """Run the engine on extracted parameters; return a normalised answer dict."""
    if not isinstance(p, dict):
        return {"parsed": False}
    try:
        op = params_to_operation(p)
        t = assess(op)
    except (InvalidOperation, KeyError, TypeError, AttributeError, ValueError) as e:
        return {"parsed": True, "calc_error": str(e)[:200], "igrc": None, "final_grc": None,
                "initial_arc": None, "residual_arc": None, "tmpr": None, "sail": None}
    return {"parsed": True, "igrc": t["igrc"], "final_grc": t["final_grc"], "initial_arc": t["initial_arc"],
            "residual_arc": t["residual_arc"], "tmpr": t["tmpr"], "sail": t["sail"]}
