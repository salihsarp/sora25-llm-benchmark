"""Synthetic UAS operation descriptions (English + Turkish) with SORA 2.5 ground truth.

Each scenario is sampled as structured parameters, validated by the deterministic engine
(sora25.assess) and rendered as a natural-language operation description in both languages.
The two renderings carry exactly the same facts (paired design).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict

from .sora25 import (Operation, Mitigation, InvalidOperation, assess, DIM_COLS, SPEED_COLS,
                     THRESH_500FT_M, FT_PER_M, operation_to_dict)

# ------------------------------------------------------------------ sampling tables
DIM_RANGES = [(0.35, 0.95), (1.1, 2.85), (3.2, 7.7), (8.5, 19.0), (21.0, 38.0)]
SPEED_RANGES = [(12.0, 24.0), (26.5, 34.0), (37.0, 72.0), (78.0, 117.0), (124.0, 190.0)]
MTOM_RANGES = [(0.9, 12.0), (6.0, 90.0), (40.0, 900.0), (400.0, 5000.0), (2000.0, 15000.0)]
BAND_RANGES = {"<5": (0.5, 4.5), "<50": (6, 45), "<500": (60, 450), "<5000": (600, 4500),
               "<50000": (6000, 45000), ">50000": (60000, 150000)}
BANDS = ["<5", "<50", "<500", "<5000", "<50000", ">50000"]

LVL_EN = {"low": "low", "medium": "medium", "high": "high"}
LVL_TR = {"low": "düşük", "medium": "orta", "high": "yüksek"}

PLACES = ["Konya", "Eskişehir", "Gebze (Kocaeli)", "Polatlı (Ankara)", "Torbalı (İzmir)", "Bursa",
          "Kayseri", "Ceyhan (Adana)", "Manavgat (Antalya)", "Samsun", "Erzurum", "Çorlu (Tekirdağ)",
          "Afyonkarahisar", "Sakarya", "Mersin", "Balıkesir"]
OPERATORS = ["Anadolu İHA Hizmetleri Ltd. Şti.", "Marmara Drone Hizmetleri A.Ş.", "Mavi Gökyüzü Havacılık A.Ş.",
             "Bozkır İHA Teknolojileri Ltd. Şti.", "Ege Hava Hizmetleri Ltd. Şti.", "Atlas İHA Operasyonları A.Ş.",
             "Kapadokya Havacılık Ltd. Şti.", "Doğu İHA Çözümleri A.Ş."]

PURPOSES = {  # key: (EN, TR)
    "powerline": ("inspection of a high-voltage power line", "yüksek gerilim enerji nakil hattı denetimi"),
    "pipeline": ("inspection of a natural-gas pipeline", "doğal gaz boru hattı denetimi"),
    "spraying": ("crop spraying", "tarımsal ilaçlama"),
    "medical": ("delivery of medical samples between two hospitals", "iki hastane arasında tıbbi numune taşıma"),
    "mapping": ("aerial mapping (photogrammetry)", "hava fotogrametrisi ile haritalama"),
    "sar": ("search-and-rescue support", "arama-kurtarma desteği"),
    "bridge": ("inspection of a motorway bridge", "otoyol köprüsü denetimi"),
    "parcel": ("parcel delivery", "paket teslimatı"),
    "fire": ("forest-fire surveillance", "orman yangını gözetimi"),
    "solar": ("thermal inspection of a solar farm", "güneş enerjisi santralinde termal inceleme"),
    "wind": ("inspection of wind turbines", "rüzgâr türbini denetimi"),
    "event": ("live filming of a large outdoor event", "büyük bir açık hava etkinliğinin canlı çekimi"),
    "testrange": ("flight testing of a new UA configuration", "yeni bir İHA konfigürasyonunun uçuş testi"),
    "airport": ("inspection of airport perimeter lighting", "havaalanı çevre aydınlatmasının denetimi"),
    "haps": ("high-altitude telecommunications relay", "yüksek irtifa telekomünikasyon aktarımı"),
    "cargo": ("cargo transport flights", "kargo taşıma uçuşları"),
}

# Table 3 descriptors, paraphrased (area descriptions only, no band labels)
QUAL = {
    "<5": ("forest and large farm parcels with roughly one small building per km²",
           "km² başına yaklaşık bir küçük yapı bulunan orman ve geniş tarım parselleri"),
    "<50": ("an area of small farms and residences on very large lots (about 16,000 m² each)",
            "küçük çiftliklerden ve çok büyük parseller (her biri yaklaşık 16.000 m²) üzerindeki konutlardan oluşan bir alan"),
    "<500": ("homes and small businesses on large lots (about 4,000 m² each)",
             "geniş parseller (her biri yaklaşık 4.000 m²) üzerindeki evler ve küçük işletmeler"),
    "<5000": ("single-family homes on small lots, apartment complexes and commercial buildings, mostly below 3-4 storeys",
              "küçük parsellerde müstakil evler, apartman siteleri ve çoğu 3-4 katı geçmeyen ticari binalar"),
    "<50000": ("mostly large multistorey buildings in the downtown area of the city",
               "şehir merkezinde ağırlıklı olarak büyük, çok katlı binalar"),
    ">50000": ("a stadium during a professional football match with a capacity crowd",
               "profesyonel bir futbol maçı sırasında tamamen dolu bir stadyum"),
}
SEGMENT_NAMES = {  # (EN, TR) by band, for multi-segment routes
    "<5": ("open steppe", "açık bozkır"), "<50": ("scattered farmland", "dağınık tarım arazisi"),
    "<500": ("a village", "bir köy"), "<5000": ("the edge of a town", "bir ilçenin kenar mahalleleri"),
    "<50000": ("a city centre", "bir şehir merkezi"), ">50000": ("a crowded festival area", "kalabalık bir festival alanı"),
}

M2_DEVICES = [("a parachute recovery system", "paraşüt kurtarma sistemi"),
              ("a frangible airframe with energy-absorbing structure", "enerji sönümleyici, kırılabilir gövde yapısı"),
              ("an airbag-and-parachute system", "hava yastığı ve paraşüt sistemi")]
M1B_EXAMPLES = [("flights only between 02:00 and 05:00 local time when streets are nearly empty",
                 "uçuşlar yalnızca sokakların neredeyse boş olduğu yerel saatle 02:00-05:00 arasında yapılır"),
                ("the route is closed to the public by the municipality during the flights",
                 "uçuşlar sırasında güzergâh belediye tarafından halka kapatılır"),
                ("flights only in the early morning (05:00-07:00), when the overflown area is nearly empty",
                 "uçuşlar yalnızca üzerinden uçulan alanın neredeyse boş olduğu sabah erken saatlerde (05:00-07:00) yapılır")]


def _fmt(x: float, lang: str, nd: int = 1) -> str:
    """Number formatting: EN 1,234.5 / TR 1.234,5."""
    if nd == 0 or (abs(x - round(x)) < 1e-9 and x >= 10):
        s = f"{int(round(x)):,}"
    else:
        s = f"{x:,.{nd}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    if lang == "tr":
        s = s.replace(",", "§").replace(".", ",").replace("§", ".")
    return s


def _round_density(d: float) -> float:
    if d < 10:
        return round(d, 1)
    mag = 10 ** (int(math.log10(d)) - 1)
    return round(d / mag) * mag


def _loguniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _mitigation(rng, target: str, ia_prob: float = 0.4, max_level: str = "high"):
    """Return (Mitigation, fmt) with robustness == target; fmt 'explicit' or 'ia'."""
    if rng.random() < ia_prob:
        levels = ["low", "medium", "high"]
        k = levels.index(target)
        # one of integrity/assurance equals target, the other >= target (and <= max_level)
        other = rng.choice(levels[k: levels.index(max_level) + 1])
        if rng.random() < 0.5:
            return Mitigation(target, other), "ia"
        return Mitigation(other, target), "ia"
    return Mitigation(target, target), "explicit"


# ------------------------------------------------------------------ sampling
def sample_scenario(rng: random.Random, sid: str) -> dict:
    for _ in range(200):
        try:
            sc = _sample_once(rng, sid)
            sc["truth"] = assess(sc["_op"])
            sc["op"] = operation_to_dict(sc.pop("_op"))
            sc["flags"] = compute_flags(sc)
            return sc
        except InvalidOperation:
            continue
    raise RuntimeError("could not sample a valid scenario")


def _sample_once(rng: random.Random, sid: str) -> dict:
    meta = {"id": sid}
    meta["special"] = None
    # -------------------------------------------------------------- airspace family first
    a = rng.random()
    atypical = a < 0.07
    above_fl600 = 0.07 <= a < 0.09
    airport_env = 0.09 <= a < 0.24
    # -------------------------------------------------------------- UA
    micro_case = rng.random() < 0.07 and not above_fl600
    if above_fl600:
        dim_col, speed_col = 4, 0
        dim = round(rng.uniform(22, 36), 1)
        speed = round(rng.uniform(15, 24), 1)
        mtom = round(rng.uniform(120, 900), 0)
        ua_type = "haps"
    elif micro_case:
        dim_col = 0
        dim = round(rng.uniform(0.18, 0.32), 2)
        mtom = rng.choice([0.249, 0.245, 0.235, 0.249])
        near = rng.random()
        if near < 0.25:          # trap: 249 g but faster than 25 m/s -> 250 g rule does NOT apply
            speed = round(rng.uniform(27, 33), 1)
            meta["special"] = "micro_fast"
        elif near < 0.40:        # trap: just above 250 g
            mtom = rng.choice([0.29, 0.32, 0.36])
            speed = round(rng.uniform(14, 22), 1)
            meta["special"] = "micro_heavy"
        else:
            speed = round(rng.uniform(13, 22), 1)
            meta["special"] = "micro"
        speed_col = 0 if speed <= 25 else 1
        ua_type = "multirotor"
    else:
        dim_col = rng.choices(range(5), weights=[40, 30, 17, 9, 4])[0]
        if rng.random() < 0.30 and dim_col < 4:      # speed drives the column
            speed_col = min(4, dim_col + rng.choice([1, 1, 2]))
        elif dim_col >= 3:
            speed_col = rng.randint(2, dim_col)        # large aircraft are not slow
        elif dim_col == 2:
            speed_col = rng.randint(1, 2)
        else:
            speed_col = rng.randint(0, dim_col)
        lo, hi = DIM_RANGES[dim_col]
        dim = round(rng.uniform(lo, hi), 2 if dim_col == 0 else 1)
        lo, hi = SPEED_RANGES[speed_col]
        speed = round(rng.uniform(lo, hi), 1)
        if speed > 38 or dim_col >= 3:
            ua_type = "fixedwing" if dim_col != 2 or rng.random() < 0.6 else "helicopter"
            if speed > 120:
                ua_type = "jet"
        elif dim_col == 2:
            ua_type = rng.choice(["helicopter", "fixedwing"])
        else:
            ua_type = rng.choice(["multirotor", "multirotor", "fixedwing", "vtol"])
        if dim_col == 0:      # small airframes: mass scales with size and type
            lo, hi = (0.6, 14 * dim) if ua_type == "multirotor" else (0.4, 4 * dim)
        elif dim_col == 1:
            lo, hi = (3.0, 35 * dim)
        else:
            lo, hi = MTOM_RANGES[dim_col]
        mtom = round(_loguniform(rng, lo, hi), 1 if dim_col <= 1 else 0)
    cruise = None
    if rng.random() < 0.4:
        cruise = round(speed * rng.uniform(0.45, 0.7), 1)   # trap: mission speed < max speed
    # -------------------------------------------------------------- ground
    controlled = rng.random() < 0.10 and not above_fl600
    densities, segs, dens_fmt = [], [], "controlled"
    if controlled:
        band = "controlled"
    else:
        band = rng.choices(BANDS, weights=[17, 17, 17, 19, 18, 12])[0]
        if above_fl600:
            band = rng.choice(["<5", "<50", "<500"])
        f = rng.random()
        if f < 0.55:
            dens_fmt = "numeric"
            densities = [_round_density(_loguniform(rng, *BAND_RANGES[band]))]
        elif f < 0.78:
            dens_fmt = "qualitative"
            densities = [_round_density(_loguniform(rng, *BAND_RANGES[band]))]  # hidden, for engine only
        else:
            dens_fmt = "multiseg"
            k = BANDS.index(band)
            lower = BANDS[:k]
            if not lower:
                dens_fmt = "numeric"
                densities = [_round_density(_loguniform(rng, *BAND_RANGES[band]))]
            else:
                others = rng.sample(lower, min(len(lower), rng.choice([1, 2])))
                seg_bands = others + [band]
                rng.shuffle(seg_bands)
                # the highest-density segment is short; the predominant one is lower
                for b in seg_bands:
                    densities.append(_round_density(_loguniform(rng, *BAND_RANGES[b])))
                segs = seg_bands
    # -------------------------------------------------------------- mitigations
    mit, mit_fmt = {}, {}
    r = rng.random()
    m1a_t = None if r < 0.40 else ("low" if r < 0.78 else "medium")
    # sheltering is only plausible over built-up areas and for UA that cannot penetrate a dwelling
    # and not where open-air assemblies of people are overflown (Annex B)
    if controlled or band in ("<5", "<50", ">50000") or mtom > 25:
        m1a_t = None
    if m1a_t:   # Annex B defines M1(A) integrity/assurance at low and medium only
        mit["m1a"], mit_fmt["m1a"] = _mitigation(rng, m1a_t, max_level="medium")
    if m1a_t != "medium" and not controlled:
        r = rng.random()
        m1b_t = None if r < 0.62 else ("medium" if r < 0.85 else ("high" if r < 0.95 else "low"))
        if m1b_t:
            mit["m1b"], mit_fmt["m1b"] = _mitigation(rng, m1b_t)
    if rng.random() < 0.25:     # Annex B defines M1(C) at low only
        mit["m1c"], mit_fmt["m1c"] = Mitigation("low", "low"), "explicit"
    r = rng.random()
    m2_t = None if r < 0.55 else ("medium" if r < 0.75 else ("high" if r < 0.90 else "low"))
    if m2_t:
        mit["m2"], mit_fmt["m2"] = _mitigation(rng, m2_t)
    # -------------------------------------------------------------- air (remaining)
    airport_bcd = airport_env and rng.random() < 0.65
    airport_cls = None
    if airport_env:
        airport_cls = rng.choice(["C", "D", "D"]) if airport_bcd else rng.choice(["E", "G", "G"])
    above500_target = rng.random() < (0.30 if not airport_env else 0.25)
    if ua_type in ("fixedwing", "jet") and not airport_env and rng.random() < 0.3:
        above500_target = True
    if dim_col >= 3 and not airport_env and rng.random() < 0.7:     # large aircraft fly high
        above500_target = True
    contingency_trap = False
    if above_fl600:
        fg, cont = 19500.0, 300.0
    elif above500_target:
        if rng.random() < 0.3:
            fg = float(rng.randrange(110, 141, 5))
            cont = float(rng.randrange(max(40, 175 - int(fg)), 81, 5))   # total >= 175 m (> 500 ft)
            contingency_trap = True
        else:
            hi_top = 301 if ua_type in ("multirotor", "vtol", "helicopter") else 1201
            fg = float(rng.randrange(180, hi_top, 20))
            cont = float(rng.choice([0, 20, 30, 50]))
    else:
        fg = float(rng.randrange(30, 121, 5))
        cont = float(rng.choice([0, 0, 5, 10, 15]))
        if fg + cont > 135:
            cont = 0.0
    tmz = (not atypical and not above_fl600 and not airport_env) and rng.random() < 0.12
    tmz_word = rng.choice(["TMZ", "TMZ", "modec"]) if tmz else None
    ctrl = (not atypical and not above_fl600 and not airport_env) and rng.random() < 0.28
    ctrl_cls = rng.choice(["C", "D", "E"]) if ctrl else None
    max_band = band if band != "controlled" else "<5"
    urban = max_band in ("<5000", "<50000", ">50000")
    # -------------------------------------------------------------- purpose / context
    high = (fg + cont) > THRESH_500FT_M and not contingency_trap
    if above_fl600:
        purpose = "haps"
    elif controlled:
        purpose = "testrange"
    elif band == ">50000" and dens_fmt != "multiseg":
        purpose = "event"
    elif airport_env:
        purpose = "airport" if dim_col <= 1 else "cargo"
    else:
        pool = ["powerline", "pipeline", "mapping", "sar", "bridge", "fire", "solar", "wind"]
        if band in ("<5000", "<50000"):
            pool = ["medical", "parcel", "bridge", "mapping", "sar"]
        if dens_fmt == "multiseg":
            pool = ["powerline", "pipeline", "medical", "parcel", "mapping"]
        if ua_type == "multirotor" and band in ("<5", "<50"):
            pool += ["spraying", "spraying"]
        if high:      # inspection tasks are not flown hundreds of metres up
            pool = ["mapping", "pipeline", "fire", "sar", "medical", "parcel"]
        if dim_col >= 3:
            pool = ["mapping", "pipeline", "fire", "sar", "cargo"]
        purpose = rng.choice(pool)
    # VLOS only where it is physically plausible (low height, slow UA, local task)
    vlos_ok = ((not above_fl600) and fg + cont <= 200 and speed <= 40 and dim_col <= 2
               and purpose not in ("medical", "parcel", "cargo", "pipeline"))
    vlos = vlos_ok and rng.random() < 0.55
    observer = (not vlos) and (not above_fl600) and rng.random() < 0.3
    op = Operation(mtom_kg=mtom, dim_m=dim, speed_ms=speed, densities=densities,
                   controlled_ground_area=controlled,
                   m1a=mit.get("m1a"), m1b=mit.get("m1b"), m1c=mit.get("m1c"), m2=mit.get("m2"),
                   atypical=atypical, above_fl600=above_fl600, airport_env=airport_env,
                   airport_class_bcd=airport_bcd, flight_geography_top_m=fg, contingency_extra_m=cont,
                   tmz_or_modec=tmz, controlled_airspace=ctrl, urban=urban, vlos=vlos,
                   airspace_observer=observer)
    meta.update({
        "ua_type": ua_type, "cruise_ms": cruise, "dens_fmt": dens_fmt, "band": band, "segs": segs,
        "mit_fmt": mit_fmt, "airport_cls": airport_cls, "tmz_word": tmz_word, "ctrl_cls": ctrl_cls,
        "contingency_trap": contingency_trap, "purpose": purpose,
        "place": rng.choice(PLACES), "operator": rng.choice(OPERATORS),
        "m2_device": rng.randrange(len(M2_DEVICES)), "m1b_example": rng.randrange(len(M1B_EXAMPLES)),
        "erp": rng.random() < 0.4, "insurance": rng.random() < 0.5, "licence": rng.random() < 0.6,
        "wind_limit": rng.choice([8, 10, 12]), "order_seed": rng.randrange(10 ** 6),
        "op_number": rng.randrange(1000, 9999),
    })
    meta["_op"] = op
    return meta


def compute_flags(sc: dict) -> dict:
    t, op = sc["truth"], sc["op"]
    mits = {k: op[k] for k in ("m1a", "m1b", "m1c", "m2") if op[k]}
    from .sora25 import MITIGATION_CREDIT, LEVELS

    def rob(m):
        return LEVELS[min(LEVELS.index(m["integrity"]), LEVELS.index(m["assurance"]))]
    na = any(MITIGATION_CREDIT[k][rob(m)] is None for k, m in mits.items())
    ia_mismatch = any(m["integrity"] != m["assurance"] for m in mits.values())
    return {
        "speed_drives_column": t["speed_column"] > t["dim_column"],
        "micro_250g": t["micro_250g"],
        "micro_trap": sc.get("special") in ("micro_fast", "micro_heavy"),
        "multiseg": sc["dens_fmt"] == "multiseg",
        "qualitative": sc["dens_fmt"] == "qualitative",
        "controlled_area": op["controlled_ground_area"],
        "cruise_distractor": sc["cruise_ms"] is not None,
        "ia_mismatch": ia_mismatch,
        "na_credit": na,
        "m1_floor": t.get("m1_floor_binding", False),
        "contingency_trap": sc["contingency_trap"],
        "vlos_reduction": t["vlos_reduction_applied"],
        "vlos_no_arc_a": t["vlos_reduction_eligible"] and t["initial_arc"] == "b",
        "airport_precedence": op["airport_env"] and t["above_500ft"],
        "airport_class_e_g": op["airport_env"] and not op["airport_class_bcd"],
        "erp_distractor": sc["erp"],
        "out_of_scope": t["sail"] == "certified",
        "m2_below_floor": bool(t.get("m1_floor") and isinstance(t.get("final_grc"), int)
                               and t["final_grc"] < t["m1_floor"]),
        "any_mitigation": bool(mits),
    }


# ------------------------------------------------------------------ rendering
UA_NAMES = {
    "multirotor": ("multirotor (quadcopter/hexacopter)", "çok rotorlu (kuadkopter/heksakopter)"),
    "fixedwing": ("fixed-wing UA", "sabit kanatlı İHA"),
    "vtol": ("hybrid VTOL fixed-wing UA", "hibrit VTOL sabit kanatlı İHA"),
    "helicopter": ("unmanned helicopter", "insansız helikopter"),
    "jet": ("jet-powered fixed-wing UA", "jet motorlu sabit kanatlı İHA"),
    "haps": ("solar-powered high-altitude platform (HAPS)", "güneş enerjili yüksek irtifa platformu (HAPS)"),
}
DIM_KIND = {
    "multirotor": ("maximum distance between blade tips", "pervane uçları arasındaki en büyük mesafe"),
    "fixedwing": ("wingspan", "kanat açıklığı"), "vtol": ("wingspan", "kanat açıklığı"),
    "jet": ("wingspan", "kanat açıklığı"), "haps": ("wingspan", "kanat açıklığı"),
    "helicopter": ("main-rotor diameter", "ana rotor çapı"),
}


def render(sc: dict, lang: str) -> str:
    assert lang in ("en", "tr")
    L = 0 if lang == "en" else 1
    op, f = sc["op"], (lambda x, nd=1: _fmt(x, lang, nd))
    rng = random.Random(sc["order_seed"])
    lines = []
    # header
    if lang == "en":
        lines.append(f"Operation reference: OP-{sc['op_number']}")
        lines.append(f"Operator and purpose: {sc['operator']} plans {PURPOSES[sc['purpose']][0]} near {sc['place']}, Türkiye.")
    else:
        lines.append(f"Operasyon referansı: OP-{sc['op_number']}")
        lines.append(f"İşletici ve amaç: {sc['operator']}, {sc['place']} yakınında {PURPOSES[sc['purpose']][1]} planlamaktadır.")
    # aircraft
    ua, dk = UA_NAMES[sc["ua_type"]][L], DIM_KIND[sc["ua_type"]][L]
    mt = op["mtom_kg"]
    mt_s = f(mt, 3) if mt < 1 else f(mt, 1 if mt < 100 else 0)
    if lang == "en":
        s = (f"Aircraft: {ua}; maximum characteristic dimension {f(op['dim_m'], 2)} m ({dk}); "
             f"maximum speed {f(op['speed_ms'])} m/s (designer-declared maximum commanded airspeed); MTOM {mt_s} kg.")
        if sc["cruise_ms"]:
            s += f" The planned cruise speed for this mission is {f(sc['cruise_ms'])} m/s."
    else:
        s = (f"Hava aracı: {ua}; azami karakteristik boyut {f(op['dim_m'], 2)} m ({dk}); "
             f"azami hız {f(op['speed_ms'])} m/s (tasarımcının beyan ettiği azami komut edilebilir hava hızı); "
             f"azami kalkış kütlesi (MTOM) {mt_s} kg.")
        if sc["cruise_ms"]:
            s += f" Bu görev için planlanan seyir hızı {f(sc['cruise_ms'])} m/s'dir."
    lines.append(s)
    # ground
    lines.append(_render_ground(sc, lang))
    # mitigations
    lines.append(_render_mitigations(sc, lang))
    # air
    lines.append(_render_air(sc, lang))
    # other information (distractors), shuffled
    other = []
    if sc["licence"]:
        other.append("The remote pilot holds an SHGM P2 remote-pilot licence." if L == 0 else
                     "Uzaktan pilot SHGM P2 İHA pilot lisansına sahiptir.")
    if sc["insurance"]:
        other.append("Third-party liability insurance is in place." if L == 0 else
                     "Üçüncü şahıs mali sorumluluk sigortası mevcuttur.")
    if sc["erp"]:
        other.append("A validated Emergency Response Plan (ERP) has been prepared." if L == 0 else
                     "Doğrulanmış bir Acil Durum Müdahale Planı (ERP) hazırlanmıştır.")
    other.append(f"Flights are cancelled if the wind exceeds {sc['wind_limit']} m/s." if L == 0 else
                 f"Rüzgâr {sc['wind_limit']} m/s'yi aşarsa uçuş iptal edilir.")
    rng.shuffle(other)
    lines.append(("Other information: " if L == 0 else "Diğer bilgiler: ") + " ".join(other))
    return "\n".join(lines)


def _render_ground(sc, lang):
    op, L = sc["op"], (0 if lang == "en" else 1)
    f = lambda x: _fmt(x, lang, 0 if x >= 10 else 1)
    if sc["dens_fmt"] == "controlled":
        return ("Ground: the whole operational volume and ground risk buffer lie in a controlled ground area: "
                "the site is fenced and guarded, and only persons involved in the operation are present."
                if L == 0 else
                "Yer: operasyon hacminin ve yer riski tampon bölgesinin tamamı kontrollü bir yer alanındadır: "
                "saha çitle çevrilidir ve koruma altındadır; yalnızca operasyona dahil kişiler bulunmaktadır.")
    if sc["dens_fmt"] == "numeric":
        d = op["densities"][0]
        return (f"Ground: according to the population density map designated by the authority, the highest population "
                f"density within the iGRC footprint (operational volume plus ground risk buffer) is about {f(d)} people/km²."
                if L == 0 else
                f"Yer: otoritenin belirlediği nüfus yoğunluğu haritasına göre iGRC izdüşümündeki (operasyon hacmi ve yer "
                f"riski tampon bölgesi) en yüksek nüfus yoğunluğu yaklaşık {f(d)} kişi/km²'dir.")
    if sc["dens_fmt"] == "qualitative":
        q = QUAL[sc["band"]][L]
        return (f"Ground: no population density map is available, so the authority accepts the SORA qualitative "
                f"descriptors. The iGRC footprint (operational volume plus ground risk buffer) lies over {q}."
                if L == 0 else
                f"Yer: nüfus yoğunluğu haritası bulunmadığından otorite SORA niteliksel tanımlayıcılarının kullanılmasını "
                f"kabul etmiştir. iGRC izdüşümü (operasyon hacmi ve yer riski tampon bölgesi) şu alanın üzerindedir: {q}.")
    # multi-segment
    parts = []
    for b, d in zip(sc["segs"], op["densities"]):
        name = SEGMENT_NAMES[b][L]
        parts.append(f"{name} (about {f(d)} people/km²)" if L == 0 else f"{name} (yaklaşık {f(d)} kişi/km²)")
    i_max = max(range(len(op["densities"])), key=lambda i: op["densities"][i])
    pre = [i for i in range(len(op["densities"])) if i != i_max]
    main = pre[0] if pre else i_max
    k = len(parts)
    if L == 0:
        return (f"Ground: the route and its ground risk buffer cross {['', '', 'two', 'three'][k]} areas on the authority's "
                f"population map: " + "; ".join(parts)
                + f". Most of the flight time is spent over {SEGMENT_NAMES[sc['segs'][main]][0]}.")
    return (f"Yer: güzergâh ve yer riski tampon bölgesi, otoritenin nüfus haritasında {['', '', 'iki', 'üç'][k]} farklı "
            f"alandan geçmektedir: " + "; ".join(parts)
            + f". Uçuş süresinin büyük bölümü {SEGMENT_NAMES[sc['segs'][main]][1]} üzerinde geçmektedir.")


def _mit_level_text(m: dict, fmt: str, lang: str) -> str:
    if fmt == "ia":
        if lang == "en":
            return f"integrity level {m['integrity']}, assurance level {m['assurance']}"
        return f"bütünlük/integrity: {LVL_TR[m['integrity']]}; güvence/assurance: {LVL_TR[m['assurance']]}"
    if lang == "en":
        return f"{m['integrity']} robustness"
    return f"sağlamlık/robustness: {LVL_TR[m['integrity']]}"


def _render_mitigations(sc, lang):
    op, L = sc["op"], (0 if lang == "en" else 1)
    items = []
    if op["m1a"]:
        t = _mit_level_text(op["m1a"], sc["mit_fmt"]["m1a"], lang)
        items.append(f"M1(A) sheltering - most people overflown are inside buildings the UA is not expected to penetrate ({t})"
                     if L == 0 else
                     f"M1(A) barınma - üzerinden uçulan kişilerin çoğu hava aracının delmesi beklenmeyen binaların içindedir ({t})")
    if op["m1b"]:
        t = _mit_level_text(op["m1b"], sc["mit_fmt"]["m1b"], lang)
        ex = M1B_EXAMPLES[sc["m1b_example"]][L]
        items.append(f"M1(B) operational restrictions - {ex} ({t})" if L == 0 else
                     f"M1(B) operasyonel kısıtlamalar - {ex} ({t})")
    if op["m1c"]:
        t = _mit_level_text(op["m1c"], sc["mit_fmt"].get("m1c", "explicit"), lang)
        items.append(f"M1(C) ground observation - observers watch the overflown area and the pilot re-routes if people appear ({t})"
                     if L == 0 else
                     f"M1(C) yer gözlemi - gözlemciler üzerinden uçulan alanı izler, insan görülürse pilot rotayı değiştirir ({t})")
    if op["m2"]:
        t = _mit_level_text(op["m2"], sc["mit_fmt"]["m2"], lang)
        dev = M2_DEVICES[sc["m2_device"]][L]
        items.append(f"M2 effects of UA impact dynamics reduced - {dev} ({t})" if L == 0 else
                     f"M2 çarpma dinamiklerinin etkilerinin azaltılması - {dev} ({t})")
    if not items:
        return ("Ground-risk mitigations: the operator claims no ground-risk mitigations." if L == 0 else
                "Yer riski azaltımları: işletici herhangi bir yer riski azaltımı talep etmemektedir.")
    head = "Ground-risk mitigations claimed by the operator: " if L == 0 else "İşleticinin talep ettiği yer riski azaltımları: "
    return head + "; ".join(items) + "."


def _render_air(sc, lang):
    op, L = sc["op"], (0 if lang == "en" else 1)
    f = lambda x: _fmt(x, lang, 0)
    parts = []
    fg, cont = op["flight_geography_top_m"], op["contingency_extra_m"]
    if op["above_fl600"]:
        parts.append("Airspace: launch and recovery are assessed in a separate application; the operation assessed here "
                     "takes place entirely above FL600." if L == 0 else
                     "Hava sahası: kalkış ve iniş ayrı bir başvuruda değerlendirilmektedir; burada değerlendirilen "
                     "operasyonun tamamı FL600 (uçuş seviyesi 600) üzerinde gerçekleşmektedir.")
    else:
        ft = fg * FT_PER_M
        if L == 0:
            s = f"Height: the flight geography extends up to {f(fg)} m AGL (about {f(round(ft, -1))} ft)"
            s += (f"; the contingency volume extends a further {f(cont)} m (about {f(round(cont * FT_PER_M, -1))} ft) "
                  f"above the flight geography." if cont > 0 else
                  "; the contingency procedures keep the UA within this height (no additional vertical contingency volume).")
        else:
            s = f"Yükseklik: uçuş coğrafyası yerden (AGL) {f(fg)} m'ye (yaklaşık {f(round(ft, -1))} ft) kadar uzanmaktadır"
            s += (f"; acil durum hacmi (contingency volume) uçuş coğrafyasının {f(cont)} m (yaklaşık "
                  f"{f(round(cont * FT_PER_M, -1))} ft) üzerine kadar uzanmaktadır." if cont > 0 else
                  "; acil durum prosedürleri hava aracını bu yüksekliğin altında tutar (ek dikey acil durum hacmi yoktur).")
        parts.append(s)
        if op["atypical"]:
            parts.append("Airspace: the entire operational volume lies inside a restricted area reserved exclusively for this "
                         "operation, which the competent authority has designated as atypical (segregated) airspace."
                         if L == 0 else
                         "Hava sahası: operasyon hacminin tamamı, yalnızca bu operasyon için ayrılmış ve yetkili otoritece "
                         "atipik (ayrılmış/segregated) hava sahası olarak belirlenmiş tahditli bir sahanın içindedir.")
        elif op["airport_env"]:
            cls = sc["airport_cls"]
            parts.append(f"Airspace: the operation takes place in the airport environment of a regional airport near "
                         f"{sc['place']}; the airspace there is class {cls}." if L == 0 else
                         f"Hava sahası: operasyon, {sc['place']} yakınındaki bir bölge havaalanının çevresinde (havaalanı "
                         f"ortamı / airport environment) gerçekleşmektedir; buradaki hava sahası {cls} sınıfıdır.")
        else:
            if op["controlled_airspace"]:
                s = (f"Airspace: the operational volume lies inside class {sc['ctrl_cls']} controlled airspace, away from any "
                     f"airport or heliport environment" if L == 0 else
                     f"Hava sahası: operasyon hacmi, herhangi bir havaalanı/heliport ortamından uzakta, {sc['ctrl_cls']} sınıfı "
                     f"kontrollü hava sahası içindedir")
            else:
                s = ("Airspace: the operational volume lies in uncontrolled (class G) airspace, away from any airport or "
                     "heliport environment" if L == 0 else
                     "Hava sahası: operasyon hacmi, herhangi bir havaalanı/heliport ortamından uzakta, kontrolsüz (G sınıfı) "
                     "hava sahasındadır")
            if op["tmz_or_modec"]:
                if sc["tmz_word"] == "modec":
                    s += ", within a Mode-C veil" if L == 0 else " ve bir Mode-C veil (Mod-C) alanı içinde kalmaktadır"
                else:
                    s += ", within a transponder mandatory zone (TMZ)" if L == 0 else " ve bir transponder zorunlu bölgesi (TMZ) içinde kalmaktadır"
            else:
                s += ", outside any TMZ or Mode-C veil" if L == 0 else "; herhangi bir TMZ veya Mode-C veil alanı içinde değildir"
            s += (", over an urban area." if op["urban"] else ", over a rural area.") if L == 0 else \
                 (". Üzerinden uçulan bölge kentseldir." if op["urban"] else ". Üzerinden uçulan bölge kırsaldır.")
            parts.append(s)
    if op["vlos"]:
        parts.append("Flight mode: the flight is conducted within visual line of sight (VLOS) of the remote pilot." if L == 0 else
                     "Uçuş şekli: uçuş, uzaktan pilotun görsel görüş hattı içinde (VLOS) gerçekleştirilir.")
    elif op["airspace_observer"]:
        parts.append("Flight mode: beyond visual line of sight (BVLOS); an airspace observer is positioned alongside the remote "
                     "pilot and communicates with them instantaneously." if L == 0 else
                     "Uçuş şekli: görüş hattı ötesi (BVLOS); uzaktan pilotun hemen yanında, onunla anlık iletişim kuran bir "
                     "hava sahası gözlemcisi bulunur.")
    else:
        parts.append("Flight mode: beyond visual line of sight (BVLOS), without airspace observers." if L == 0 else
                     "Uçuş şekli: hava sahası gözlemcisi olmadan görüş hattı ötesi (BVLOS).")
    return " ".join(parts)


def generate(n: int, seed: int = 2026) -> list:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        sc = sample_scenario(rng, f"S{i:05d}")
        sc["text_en"] = render(sc, "en")
        sc["text_tr"] = render(sc, "tr")
        out.append(sc)
    return out


def save_jsonl(scs: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for sc in scs:
            fh.write(json.dumps(sc, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
