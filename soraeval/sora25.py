"""Deterministic ground-truth engine for JARUS SORA edition 2.5.

Source: JARUS guidelines on Specific Operations Risk Assessment (SORA), Main Body, edition 2.5,
JAR-DEL-SRM-SORA-MB-2.5, public release 13 May 2024 (page numbers below refer to that document);
Annex B edition 2.5 (JAR_doc_27) for the M1(A)/M1(B) combination rule.

Only the steps whose outcome is fully determined by the operation description are implemented:
Step 2 (iGRC), Step 3 (final GRC), Step 4 (initial ARC), Step 5 (residual ARC, Main Body VLOS /
airspace-observer reduction only - no Annex C density claims), Step 6 (TMPR), Step 7 (SAIL) and
Step 9 (OSO robustness from SAIL). Step 8 (containment) is out of scope for this benchmark.

The rules are encoded as facts (numbers) taken from the tables; no JARUS text is reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------- Step 2 - Table 2 (p. 34)
DIM_COLS = (1.0, 3.0, 8.0, 20.0, 40.0)          # max characteristic dimension, m
SPEED_COLS = (25.0, 35.0, 75.0, 120.0, 200.0)    # max speed, m/s
POP_BANDS = ("controlled", "<5", "<50", "<500", "<5000", "<50000", ">50000")
IGRC_TABLE = {
    "controlled": (1, 1, 2, 3, 3),
    "<5":         (2, 3, 4, 5, 6),
    "<50":        (3, 4, 5, 6, 7),
    "<500":       (4, 5, 6, 7, 8),
    "<5000":      (5, 6, 7, 8, 9),
    "<50000":     (6, 7, 8, 9, 10),
    ">50000":     (7, 8, None, None, None),       # grey cells: not part of SORA
}
BAND_UPPER = {"<5": 5, "<50": 50, "<500": 500, "<5000": 5000, "<50000": 50000}

# ---------------------------------------------------------------- Step 3 - Table 5 (p. 38)
# None = "N/A" in Table 5 (no credit at that robustness level).
MITIGATION_CREDIT = {
    "m1a": {"low": -1, "medium": -2, "high": None},
    "m1b": {"low": None, "medium": -1, "high": -2},
    "m1c": {"low": -1, "medium": None, "high": None},
    "m2":  {"low": None, "medium": -1, "high": -2},
}
LEVELS = ("low", "medium", "high")

# ---------------------------------------------------------------- Step 6 - Table 6 (p. 45)
TMPR = {"a": "none", "b": "low", "c": "medium", "d": "high"}

# ---------------------------------------------------------------- Step 7 - Table 7 (p. 47)
SAIL_TABLE = {  # final GRC row -> SAIL for residual ARC a, b, c, d
    2: ("I", "II", "IV", "VI"),     # row "<=2"
    3: ("II", "II", "IV", "VI"),
    4: ("III", "III", "IV", "VI"),
    5: ("IV", "IV", "IV", "VI"),
    6: ("V", "V", "V", "VI"),
    7: ("VI", "VI", "VI", "VI"),
}
SAIL_ORDER = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "certified": 7}
ARC_ORDER = {"a": 1, "b": 2, "c": 3, "d": 4}

# ---------------------------------------------------------------- Step 9 - Table 14 (p. 54)
OSO_TABLE = {  # robustness for SAIL I..VI
    "OSO#01": ("NR", "L", "M", "H", "H", "H"),
    "OSO#02": ("NR", "NR", "L", "M", "H", "H"),
    "OSO#03": ("L", "L", "M", "M", "H", "H"),
    "OSO#04": ("NR", "NR", "NR", "L", "M", "H"),
    "OSO#05": ("NR", "NR", "L", "M", "H", "H"),
    "OSO#06": ("NR", "L", "L", "M", "H", "H"),
    "OSO#07": ("L", "L", "M", "M", "H", "H"),
    "OSO#08": ("L", "M", "H", "H", "H", "H"),
    "OSO#09": ("L", "L", "M", "M", "H", "H"),
    "OSO#13": ("L", "L", "M", "H", "H", "H"),
    "OSO#16": ("L", "L", "M", "M", "H", "H"),
    "OSO#17": ("L", "L", "M", "M", "H", "H"),
    "OSO#18": ("NR", "NR", "L", "M", "H", "H"),
    "OSO#19": ("NR", "NR", "L", "M", "M", "H"),
    "OSO#20": ("NR", "L", "L", "M", "M", "H"),
    "OSO#23": ("L", "L", "M", "M", "H", "H"),
    "OSO#24": ("NR", "NR", "M", "H", "H", "H"),
}

FT_PER_M = 3.28084
THRESH_500FT_M = 500 / FT_PER_M  # 152.4 m


class InvalidOperation(ValueError):
    """Raised for inputs the benchmark never generates (ambiguous or not covered by SORA)."""


@dataclass
class Mitigation:
    """A claimed ground-risk mitigation. Robustness = lower of integrity and assurance (p. 26, Table 1)."""
    integrity: str
    assurance: str

    @property
    def robustness(self) -> str:
        return LEVELS[min(LEVELS.index(self.integrity), LEVELS.index(self.assurance))]


@dataclass
class Operation:
    # UA
    mtom_kg: float
    dim_m: float
    speed_ms: float
    # ground: densities (people/km2) of every footprint segment; None entries = controlled ground area
    densities: list = field(default_factory=list)
    controlled_ground_area: bool = False
    # ground-risk mitigations (None = not claimed)
    m1a: Optional[Mitigation] = None
    m1b: Optional[Mitigation] = None
    m1c: Optional[Mitigation] = None
    m2: Optional[Mitigation] = None
    # air
    atypical: bool = False
    above_fl600: bool = False
    airport_env: bool = False
    airport_class_bcd: bool = False
    flight_geography_top_m: float = 100.0   # AGL
    contingency_extra_m: float = 0.0          # contingency volume above flight geography
    tmz_or_modec: bool = False
    controlled_airspace: bool = False
    urban: bool = False
    vlos: bool = False
    airspace_observer: bool = False           # BVLOS pilot supported by observer alongside


# ------------------------------------------------------------------------------ helpers
def pop_band(density: Optional[float]) -> str:
    if density is None:
        return "controlled"
    for band, upper in BAND_UPPER.items():
        if density < upper:
            return band
    if density == 50000:
        raise InvalidOperation("density exactly 50,000 is not covered by Table 2 bands")
    return ">50000"


def column_index(dim_m: float, speed_ms: float) -> int:
    """Left-most column matching BOTH dimension and speed (p. 35, 'Identification of the iGRC' (a))."""
    for i, (d, s) in enumerate(zip(DIM_COLS, SPEED_COLS)):
        if dim_m <= d and speed_ms <= s:
            return i
    raise InvalidOperation("UA larger than 40 m or faster than 200 m/s: Annex F applies")


def mitigation_credit(kind: str, m: Optional[Mitigation]) -> int:
    if m is None:
        return 0
    c = MITIGATION_CREDIT[kind][m.robustness]
    return 0 if c is None else c


def sail_lookup(final_grc: int, residual_arc: str) -> str:
    if final_grc > 7:
        return "certified"
    row = SAIL_TABLE[max(final_grc, 2)]
    return row["abcd".index(residual_arc)]


# ------------------------------------------------------------------------------ steps
def step2_igrc(op: Operation) -> dict:
    col = column_index(op.dim_m, op.speed_ms)
    dim_col = next(i for i, d in enumerate(DIM_COLS) if op.dim_m <= d)
    speed_col = next(i for i, s in enumerate(SPEED_COLS) if op.speed_ms <= s)
    if op.controlled_ground_area:
        band = "controlled"
    else:
        if not op.densities:
            raise InvalidOperation("no population density given")
        band = pop_band(max(op.densities))       # highest density in the footprint (p. 35)
    micro = op.mtom_kg <= 0.25 and op.speed_ms <= 25.0
    if micro:
        igrc = 1                                   # 250 g rule (p. 34)
    else:
        igrc = IGRC_TABLE[band][col]
    return {"column": col, "dim_column": dim_col, "speed_column": speed_col, "pop_band": band,
            "micro_250g": micro, "igrc": igrc}


def step3_final_grc(op: Operation, igrc: Optional[int], col: int) -> dict:
    if igrc is None:
        return {"m1_credit": None, "m2_credit": None, "final_grc": None, "m1_floor_binding": False}
    if op.m1a is not None and op.m1a.robustness == "medium" and op.m1b is not None:
        raise InvalidOperation("M1(A) medium cannot be combined with M1(B) (Annex B)")
    for kind in ("m1a", "m1b", "m1c", "m2"):
        m = getattr(op, kind)
        if m is not None and m.robustness == "high" and kind == "m1a":
            raise InvalidOperation("M1(A) high is N/A in Table 5 (ambiguous)")
        if m is not None and m.robustness in ("medium", "high") and kind == "m1c":
            raise InvalidOperation("M1(C) above low is N/A in Table 5 (ambiguous)")
    m1 = mitigation_credit("m1a", op.m1a) + mitigation_credit("m1b", op.m1b) + mitigation_credit("m1c", op.m1c)
    m2 = mitigation_credit("m2", op.m2)
    col_floor = IGRC_TABLE["controlled"][col]
    after_m1 = igrc + m1
    floor_binding = m1 != 0 and after_m1 < min(col_floor, igrc)
    after_m1 = max(after_m1, min(col_floor, igrc))   # M1 floor: lowest value of the column (p. 39 (f))
    final = after_m1 + m2
    if final < 1:
        raise InvalidOperation("final GRC below 1 (not generated)")
    return {"m1_credit": m1, "m2_credit": m2, "grc_after_m1": after_m1, "final_grc": final,
            "m1_floor_binding": floor_binding, "m1_floor": col_floor}


def ops_vertical_limit_m(op: Operation) -> float:
    return op.flight_geography_top_m + op.contingency_extra_m


def step4_initial_arc(op: Operation) -> dict:
    """Figure 6 decision tree (p. 41). AEC labels follow Annex C and are for analysis only."""
    top = ops_vertical_limit_m(op)
    above500 = top > THRESH_500FT_M
    if op.atypical:
        arc, aec = "a", 12
    elif op.above_fl600:
        arc, aec = "b", 11
    elif op.airport_env:
        arc, aec = ("d", 1) if op.airport_class_bcd else ("c", 6)
    elif above500:
        if op.tmz_or_modec:
            arc, aec = "d", 2
        elif op.controlled_airspace:
            arc, aec = "d", 3
        elif op.urban:
            arc, aec = "c", 4
        else:
            arc, aec = "c", 5
    else:
        if op.tmz_or_modec:
            arc, aec = "c", 7
        elif op.controlled_airspace:
            arc, aec = "c", 8
        elif op.urban:
            arc, aec = "c", 9
        else:
            arc, aec = "b", 10
    return {"initial_arc": arc, "aec": aec, "ops_top_m": top, "above_500ft": above500}


def step5_residual_arc(op: Operation, initial_arc: str) -> dict:
    """Main Body 4.5.4 (p. 43-44): VLOS, or BVLOS with an airspace observer alongside the pilot,
    lowers the initial ARC by one class but never to ARC-a. No Annex C claims are made."""
    eligible = op.vlos or op.airspace_observer
    if eligible and initial_arc in ("c", "d"):
        residual = "bcd"["cd".index(initial_arc)]
        reduced = True
    else:
        residual = initial_arc
        reduced = False
    return {"residual_arc": residual, "vlos_reduction_eligible": eligible, "vlos_reduction_applied": reduced}


def step6_tmpr(op: Operation, residual_arc: str) -> str:
    """Table 6 for BVLOS; VLOS is an acceptable tactical mitigation for all ARCs (p. 45, 4.6.4 (a),(c))."""
    if op.vlos:
        return "none"
    return TMPR[residual_arc]


def assess(op: Operation) -> dict:
    """Full ground truth for one operation."""
    g = step2_igrc(op)
    col = g["column"]
    igrc_val = g["igrc"]
    out_of_scope_igrc = igrc_val is None
    f = step3_final_grc(op, igrc_val, col)
    a = step4_initial_arc(op)
    r = step5_residual_arc(op, a["initial_arc"])
    if out_of_scope_igrc:
        sail = "certified"
        final = None
    else:
        final = f["final_grc"]
        sail = sail_lookup(final, r["residual_arc"])
    tmpr = step6_tmpr(op, r["residual_arc"])
    oso = None
    if sail != "certified":
        k = SAIL_ORDER[sail] - 1
        oso = {o: v[k] for o, v in OSO_TABLE.items()}
    robust = {k: (getattr(op, k).robustness if getattr(op, k) else "none") for k in ("m1a", "m1b", "m1c", "m2")}
    return {
        **g, "igrc": "out_of_scope" if out_of_scope_igrc else igrc_val,
        **{k: v for k, v in f.items() if k != "final_grc"}, "final_grc": final,
        **a, **r, "tmpr": tmpr, "sail": sail, "sail_ord": SAIL_ORDER[sail], "oso": oso,
        "robustness": robust,
    }


def operation_from_dict(d: dict) -> Operation:
    d = dict(d)
    for k in ("m1a", "m1b", "m1c", "m2"):
        if d.get(k):
            d[k] = Mitigation(**d[k])
    return Operation(**d)


def operation_to_dict(op: Operation) -> dict:
    return asdict(op)
