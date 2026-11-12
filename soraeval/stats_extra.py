"""Extra statistics and figures (needs pandas, statsmodels, matplotlib).

  python -m soraeval.stats_extra --results results/main_*.jsonl --stability results/stability_haiku55.jsonl --outdir results/main_extra

Outputs: gee_<model>_<condition>.csv (odds ratios of SAIL error / under-classification on trap flags + language, scenario
clusters), kappa.txt (Fleiss kappa of SAIL across repeats), bootstrap.csv (between-model differences on shared scenarios),
fig1_under.png/pdf, fig2_traps.png/pdf, fig3_sources.png/pdf.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from .analyze import grade, wilson, TRAP_STEP
from .generator import load_jsonl

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# names exactly as in the JSONL "model" field; models missing here are dropped from the figures and the bootstrap
MODEL_ORDER = ["claude-haiku-4-5", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
               "gpt-oss:20b", "mistral-small3.2:24b", "gemma4:31b", "qwen3.8:27b"]
MODEL_LABEL = {"claude-haiku-4-5": "Haiku 4.5", "claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5",
               "claude-opus-5-5": "Opus 5.5", "gpt-oss:20b": "gpt-oss 20B", "mistral-small3.2:24b": "Mistral Small 3.2",
               "gemma4:31b": "Gemma 4 31B", "qwen3.8:27b": "Qwen3.8 27B"}
# two-line labels for tight axes
MODEL_SHORT = {k: v.replace(" ", "\n", 1) if len(v) > 9 else v for k, v in MODEL_LABEL.items()}
N_CLAUDE = 4  # first N_CLAUDE entries of MODEL_ORDER are API models; a divider separates them from open-weight models


def _divider(ax, models, step=1.0, offset=0.0):
    """Dotted vertical line between the last API model and the first open-weight model, if both are present."""
    api = [m for m in models if m in MODEL_ORDER[:N_CLAUDE]]
    if api and len(api) < len(models):
        ax.axvline(offset + step * len(api) - 0.5, color=INK2, linestyle=":", linewidth=0.8)
COND_ORDER = ["closed_book", "reference", "extract"]
COND_LABEL = {"closed_book": "Closed book", "reference": "Rule card", "extract": "Extract + calculator"}
# categorical slots 1-3 of the validated reference palette (all-pairs safe), text inks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_graded(result_paths, scen_path):
    scs = {s["id"]: s for s in load_jsonl(scen_path)}
    recs = {}
    for p in result_paths:
        for r in load_jsonl(p):
            if r.get("error") and r["key"] in recs:
                continue
            recs[r["key"]] = r
    df = pd.DataFrame([grade(r, scs[r["sid"]]) for r in recs.values() if r["sid"] in scs])
    unknown = sorted(set(df.model) - set(MODEL_ORDER)) if len(df) else []
    if unknown:
        warnings.warn(f"models not in MODEL_ORDER (left out of figures and bootstrap): {unknown}")
    return df


# ------------------------------------------------------------------ GEE
def run_gee(df, outdir):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    flags = [f"flag_{t}" for t in TRAP_STEP if f"flag_{t}" in df.columns and t != "out_of_scope"]
    rows = []
    for (m, c), d in df.groupby(["model", "condition"]):
        d = d.copy()
        d["tr"] = (d["lang"] == "tr").astype(int)
        d["sail_wrong"] = (~d["ok_sail"].astype(bool)).astype(int)
        d["under"] = d["sail_under"].astype(int)
        use = [f for f in flags if d[f].astype(int).sum() >= 10 and d[f].astype(int).sum() <= len(d) - 10]
        for outcome in ("sail_wrong", "under"):
            if d[outcome].sum() < 15:
                continue
            for f in use:
                d[f] = d[f].astype(int)
            formula = f"{outcome} ~ tr + " + " + ".join(use)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = smf.gee(formula, groups="sid", data=d, family=sm.families.Binomial(),
                                  cov_struct=sm.cov_struct.Exchangeable()).fit()
                ci = res.conf_int()
                for name in res.params.index:
                    if name == "Intercept":
                        continue
                    # separation check: zero events among rows with the term (or without it) gives an absurd OR
                    on = d[name].astype(int) == 1
                    ev1, ev0 = int(d.loc[on, outcome].sum()), int(d.loc[~on, outcome].sum())
                    rows.append({"model": m, "condition": c, "outcome": outcome, "term": name.replace("flag_", ""),
                                 "OR": float(np.exp(res.params[name])), "lo": float(np.exp(ci.loc[name, 0])),
                                 "hi": float(np.exp(ci.loc[name, 1])), "p": float(res.pvalues[name]),
                                 "n": len(d), "events": int(d[outcome].sum()),
                                 "events_with_term": ev1, "events_without_term": ev0,
                                 "no_events": "no events" if (ev1 == 0 or ev0 == 0) else ""})
            except Exception as e:  # noqa
                rows.append({"model": m, "condition": c, "outcome": outcome, "term": "FAILED", "OR": np.nan,
                             "lo": np.nan, "hi": np.nan, "p": np.nan, "n": len(d), "events": int(d[outcome].sum()),
                             "note": str(e)[:120]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, "gee.csv"), index=False)
    return out


# ------------------------------------------------------------------ stability
def fleiss_kappa_sail(stab_df):
    from statsmodels.stats.inter_rater import fleiss_kappa
    cats = ["I", "II", "III", "IV", "V", "VI", "certified", "missing"]
    table = []
    agree_all = 0
    for sid, d in stab_df.groupby("sid"):
        vals = [v if isinstance(v, str) and v in cats else "missing" for v in d["pred_sail"]]
        if len(vals) < 3:
            continue
        table.append([vals.count(c) for c in cats])
        agree_all += int(len(set(vals)) == 1)
    table = np.array(table)
    return float(fleiss_kappa(table)), agree_all, len(table)


# ------------------------------------------------------------------ bootstrap
def bootstrap_models(df, outdir, B=2000, seed=1, models_subset=None, fname="bootstrap.csv"):
    """Paired bootstrap of between-model differences on the scenarios shared by all models in `models_subset`
    (default: every model present). One random stream per call, seeded with `seed`."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in COND_ORDER:
        for lang in ("en", "tr"):
            d = df[(df.condition == c) & (df.lang == lang)]
            models = [m for m in MODEL_ORDER if m in set(d.model) and (models_subset is None or m in models_subset)]
            shared = set.intersection(*[set(d[d.model == m].sid) for m in models]) if models else set()
            shared = sorted(shared)
            if len(shared) < 20:
                continue
            piv = {m: d[d.model == m].set_index("sid").loc[shared] for m in models}
            for i, a in enumerate(models):
                for b in models[i + 1:]:
                    for metric in ("sail_under", "ok_sail"):
                        xa, xb = piv[a][metric].astype(float).values, piv[b][metric].astype(float).values
                        diff = xa.mean() - xb.mean()
                        idx = rng.integers(0, len(shared), size=(B, len(shared)))
                        boots = xa[idx].mean(1) - xb[idx].mean(1)
                        rows.append({"condition": c, "lang": lang, "metric": metric, "a": a, "b": b, "n": len(shared),
                                     "diff": diff, "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, fname), index=False)
    return out


# the 500 scenarios (S00000-S00499) assessed by Haiku 5.5, Sonnet 5.5 and the four open models
MODELS_500 = ["claude-haiku-5-5", "claude-sonnet-5-5", "gpt-oss:20b", "mistral-small3.2:24b", "gemma4:31b", "qwen3.8:27b"]


# ------------------------------------------------------------------ figures
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_under(df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharey=True)
    w = 0.26
    for ax, lang in zip(axes, ("en", "tr")):
        for j, c in enumerate(COND_ORDER):
            xs, ys, lo, hi = [], [], [], []
            for i, m in enumerate(models):
                d = df[(df.model == m) & (df.condition == c) & (df.lang == lang)]
                if not len(d):
                    continue
                k, n = int(d.sail_under.sum()), len(d)
                l, h = wilson(k, n)
                xs.append(i + (j - 1) * w)
                ys.append(100 * k / n)
                lo.append(max(0.0, 100 * k / n - 100 * l))
                hi.append(max(0.0, 100 * h - 100 * k / n))
            ax.bar(xs, ys, width=w - 0.03, color=SERIES[j], label=COND_LABEL[c], edgecolor="white", linewidth=1)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=8, color=INK, linespacing=1.0)
        ax.set_title("English" if lang == "en" else "Turkish", fontsize=9, color=INK)
        ax.set_ylabel("SAIL under-classified (%)", fontsize=8, color=INK)
        _divider(ax, models)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig1_under.{ext}"), dpi=300)
    plt.close(fig)


def fig_traps(df, outdir, condition="closed_book"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    traps = [t for t in TRAP_STEP if f"flag_{t}" in df.columns and t not in ("m2_below_floor",)]
    mat = np.full((len(traps), len(models)), np.nan)
    for i, t in enumerate(traps):
        for j, m in enumerate(models):
            d = df[(df.model == m) & (df.condition == condition)]
            a = d[d[f"flag_{t}"].astype(bool)]
            if len(a) >= 5:
                mat[i, j] = 100 * (1 - a[f"ok_{TRAP_STEP[t]}"].astype(bool).mean())
    fig, ax = plt.subplots(figsize=(max(4.6, 1.2 + 0.75 * len(models)), 5.4))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(traps)):
        for j in range(len(models)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > 55 else INK)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=7.5, rotation=0)
    _divider(ax, models)
    ax.set_yticks(range(len(traps)))
    ax.set_yticklabels([f"{t.replace('_', ' ')} ({TRAP_STEP[t].replace('_', ' ')})" for t in traps], fontsize=7)
    ax.set_title(f"Error of the targeted step when the trap is present (%)\n{COND_LABEL[condition]}", fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig2_traps_{condition}.{ext}"), dpi=300)
    plt.close(fig)


def fig_sources(df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    srcs = ["ground", "air", "ground+air", "lookup", "unparsed"]
    labels, data = [], {s: [] for s in srcs}
    for m in models:
        for c in COND_ORDER:
            d = df[(df.model == m) & (df.condition == c)]
            if not len(d):
                continue
            labels.append({"closed_book": "Closed", "reference": "Card", "extract": "Calc"}[c])
            for s in srcs:
                data[s].append(100 * (d.sail_err_source == s).mean())
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bottom = np.zeros(len(labels))
    for k, s in enumerate(srcs):
        ax.bar(range(len(labels)), data[s], bottom=bottom, color=SERIES[k], label=s, width=0.7,
               edgecolor="white", linewidth=1)
        bottom += np.array(data[s])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
    for i, m in enumerate(models):          # model name under each group of three bars
        ax.text(3 * i + 1, -0.25, MODEL_SHORT[m], transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color=INK, linespacing=1.0)
    _divider(ax, models, step=3)
    ax.set_ylabel("Episodes with a wrong SAIL (%)", fontsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig3_sources.{ext}"), dpi=300)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=os.path.join(HERE, "data", "scenarios.jsonl"))
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--stability", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    df = load_graded(args.results, args.scenarios)
    print(f"{len(df)} graded episodes")
    gee = run_gee(df, args.outdir)
    sig = gee[(gee.p < 0.05) & (gee.term != "FAILED")].sort_values(["model", "condition", "outcome", "p"])
    print(sig[["model", "condition", "outcome", "term", "OR", "lo", "hi", "p"]].to_string(index=False))
    bs = bootstrap_models(df, args.outdir)
    print(bs.to_string(index=False))
    if set(MODELS_500) <= set(df.model):
        bs5 = bootstrap_models(df, args.outdir, models_subset=MODELS_500, fname="bootstrap_500.csv")
        print(f"bootstrap_500.csv: {len(bs5)} rows, n = {sorted(set(bs5.n))}")
    if args.stability:
        st = load_graded([args.stability], args.scenarios)
        k, agree, n = fleiss_kappa_sail(st)
        msg = (f"Fleiss kappa (SAIL, 3 repeats, {n} scenarios): {k:.3f}; all three repeats identical in {agree}/{n}; "
               f"SAIL exact per repeat: " + ", ".join(f"{100 * st[st.rep == r].ok_sail.mean():.1f}%" for r in sorted(st.rep.unique()))
               + "; under: " + ", ".join(f"{100 * st[st.rep == r].sail_under.mean():.1f}%" for r in sorted(st.rep.unique())))
        print(msg)
        open(os.path.join(args.outdir, "kappa.txt"), "w").write(msg + "\n")
    fig_under(df, args.outdir)
    for c in ("closed_book", "reference"):
        fig_traps(df, args.outdir, c)
    fig_sources(df, args.outdir)
    print(f"figures written to {args.outdir}")


if __name__ == "__main__":
    main()
