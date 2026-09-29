# utils.py
"""Shared helpers for the ablation statistics scripts."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm, t, wilcoxon
from statsmodels.stats.multitest import multipletests


N_BOOT = 10000
SEED = 1337

# Grouped table header: metric -> (group label, sub label)
METRIC_HEADER = {
    "hd_wall":   ("dHD (mm)", "Wall"),
    "hd_lumen":  ("dHD (mm)", "Lumen"),
    "acd_wall":  ("ACSD (mm)", "Wall"),
    "acd_lumen": ("ACSD (mm)", "Lumen"),
    "cl_HD":     ("clHD", ""),
    "cl_sens":   ("clSens", ""),
    "lumen_bg":  ("LCR", ""),
    "dsc_lumen": ("DSC", "Lumen"),
    "dsc_wall":  ("DSC", "Wall"),
}
BOLDFMT = r"\textbf{{{}}}"

# --------------------------------------------------------------------------- #
# clustering unit
# --------------------------------------------------------------------------- #
def add_subject(df: pd.DataFrame, pattern: str = r"^(?P<subject_id>[^_]+)",
                col: str = "case_id") -> pd.DataFrame:
    out = df.copy()
    out["subject_id"] = out[col].str.extract(pattern)
    if out["subject_id"].isna().any():
        bad = out.loc[out["subject_id"].isna(), col].unique()[:5]
        raise ValueError(f"cannot parse subject from {list(bad)!r}")
    n_s, n_c = out["subject_id"].nunique(), out[col].nunique()
    print(f"  [utils] clustering unit: {n_s} subjects / {n_c} cases")
    return out

# Metrics defined once per case rather than per cross-section.
CASE_WISE_METRICS = ("cl_HD", "cl_sens", "lumen_bg")

METRIC_MAP = {
    ("hausdorff_distances", "Lumen"): "hd_lumen",
    ("hausdorff_distances", "Wall"): "hd_wall",
    ("average_contour_distances", "Lumen"): "acd_lumen",
    ("average_contour_distances", "Wall"): "acd_wall",
    ("centerline_sensitivity", None): "cl_sens",
    ("centerline_HD", None): "cl_HD",
    ("lumen_background_percentage", None): "lumen_bg",
    ("dice_coefficients", "Lumen"): "dsc_lumen",
    ("dice_coefficients", "Wall"): "dsc_wall",
    ("hausdorff_distances_95", "Lumen"): "hd95_lumen",
    ("hausdorff_distances_95", "Wall"): "hd95_wall",
    ("hausdorff_distances_95", "Combined"): "hd95_combined",
    ("average_surface_distances", "Lumen"): "assd_lumen",
    ("average_surface_distances", "Wall"): "assd_wall",
    ("average_surface_distances", "Combined"): "assd_combined",
    ("dice_coefficients", "Combined"): "dsc_combined"

}
METRICS = list(METRIC_MAP.values())
LOWER_IS_BETTER = {"hd_lumen": True, "hd95_lumen": True,  "hd95_wall": True, "hd_wall": True, "acd_lumen": True,
                   "acd_wall": True, "cl_sens": False, "cl_HD": True, "lumen_bg": True, "dsc_lumen": False, "dsc_wall": False, "hd_lumen_median": True, "hd_wall_median": True, "acd_lumen_median": True,
                   "acd_wall_median": True, "cl_sens_median": False, "cl_HD_median": True, "lumen_bg_median": True, "dsc_lumen_median": False, "dsc_wall_median": False}
TABLE_METRICS = ["hd_wall", "hd_lumen", "acd_wall", "acd_lumen", "cl_HD", "lumen_bg", "dsc_lumen", "dsc_wall"]

ZERO_METHOD = "pratt"
ALPHA = 0.05
DEC = 3

MARK_SIG = {"worse": r"$^{\ast}$",      # reference significantly better
            "better": r"$^{\dagger}$",  # listed configuration significantly better
            "": ""}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> pd.DataFrame:
    """One JSON file -> tidy frame (config, case_id, cs_id, metrics).

    Metrics absent from the file (e.g. no wall segmentation) are added as NaN
    columns, so all configs share the same schema.
    """
    path = Path(path)
    raw = pd.json_normalize(json.loads(path.read_text()), sep="__")
    rename = {"__".join(k for k in key if k): name for key, name in METRIC_MAP.items()}
    df = raw.rename(columns=rename)

    missing = [k for k in METRICS if k not in df.columns]
    if missing:
        print(f"  [utils] {path.stem}: metric(s) not in file, filled with NaN: {missing}")
        df = df.assign(**{k: np.nan for k in missing})

    df[["case_id", "cs_id"]] = df["identifier"].str.split("__", n=1, expand=True)
    df["config"] = path.stem
    return df[["config", "case_id", "cs_id", *METRICS]]


def load_all(folder, configs=None) -> pd.DataFrame:
    """All JSON files in `folder`, or exactly `configs` (ordered categorical)."""
    folder = Path(folder)
    if configs is None:
        paths = sorted(folder.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"no JSON files in {folder}")
        categories, ordered = [p.stem for p in paths], False
    else:
        paths = [folder / f"{c}.json" for c in configs]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing JSON file(s): {missing}")
        categories, ordered = list(configs), True

    df = pd.concat([load_config(p) for p in paths], ignore_index=True)
    df["config"] = pd.Categorical(df["config"], categories=categories, ordered=ordered)
    return df

def available_metrics(df: pd.DataFrame, metrics=None) -> list:
    """Metrics that contain at least one non-NaN value (order preserved)."""
    metrics = METRICS if metrics is None else list(metrics)
    keep = [k for k in metrics if k in df.columns and df[k].notna().any()]
    dropped = [k for k in metrics if k not in keep]
    if dropped:
        print(f"  [utils] dropping all-NaN metric(s): {dropped}")
    return keep


# --------------------------------------------------------------------------- #
# two-stage aggregation:  cross-sections -> case -> subject (unit)
# --------------------------------------------------------------------------- #
def unit_level(df: pd.DataFrame, metrics=None, unit: str = "subject_id",
               case: str = "case_id") -> pd.DataFrame:
    """Mean per (config, unit) obtained in two unweighted stages.

    1) mean over cross-sections within each case
    2) mean over cases within each unit (subject)

    Every case therefore counts the same regardless of how many cross-sections
    it has, and every subject counts the same regardless of how many cases it
    contributes.  Case-wise metrics need no special treatment: they are constant
    (or NaN) within a case, so stage 1 reproduces the case value.
    """
    metrics = available_metrics(df, metrics)
    if unit == case:
        return df.groupby(["config", unit], observed=True)[metrics].mean()
    per_case = df.groupby(["config", unit, case], observed=True)[metrics].mean()
    return per_case.groupby(level=["config", unit], observed=True).mean()

# --------------------------------------------------------------------------- #
# case-level aggregation (for the paired Wilcoxon tests)
# --------------------------------------------------------------------------- #
def case_level(df: pd.DataFrame, metrics=None, unit: str = "subject_id",
               case: str = "case_id") -> pd.DataFrame:
    """Backwards-compatible alias of `unit_level` (now two-stage)."""
    return unit_level(df, metrics, unit=unit, case=case)

# --------------------------------------------------------------------------- #
# cluster bootstrap: mean over cross-sections, resampling units
# --------------------------------------------------------------------------- #
def cluster_frames(df, metrics, case_wise=CASE_WISE_METRICS, unit="case_id"):
    """Per (config, unit) sum and count, so that a mean over cross-sections is
    sum(sums) / sum(counts).  Case-wise metrics are collapsed to one value."""
    metrics = [k for k in metrics if k in df.columns]
    g = df.groupby(["config", unit], observed=True)[metrics]
    total, count = g.sum(min_count=1), g.count().astype(float)

    cw = [k for k in case_wise if k in metrics]
    if cw:
        total[cw] = total[cw] / count[cw].replace(0.0, np.nan)   # case mean
        count[cw] = (count[cw] > 0).astype(float)                # weight 1 per case
    return total.fillna(0.0), count.fillna(0.0)

def bootstrap_weights(units, n_boot=N_BOOT, seed=SEED) -> np.ndarray:
    """(n_boot, n_units) multiplicities of a bootstrap over units."""
    rng = np.random.default_rng(seed)
    n = len(units)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)

def _unit_matrix(level, cfg, units, metrics):
    """Subject-level values of one config as (values, availability mask)."""
    v = level.xs(cfg, level="config").reindex(units)[metrics].to_numpy(float)
    m = np.isfinite(v)
    return np.where(m, v, 0.0), m.astype(float)

def _unit_means(vals, mask, w) -> np.ndarray:
    """Unweighted mean over (resampled) units, ignoring missing values."""
    num, den = w @ vals, w @ mask
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)

def _means(total, count, units, metrics, w) -> np.ndarray:
    a = np.nan_to_num(total.reindex(units)[metrics].to_numpy(float))
    c = np.nan_to_num(count.reindex(units)[metrics].to_numpy(float))
    num, den = w @ a, w @ c
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _config_order(df):
    c = df["config"]
    return list(c.cat.categories) if isinstance(c.dtype, pd.CategoricalDtype) \
        else sorted(c.unique())


def bootstrap_summary(df, metrics=None, case_wise=None, unit="subject_id",
                      case="case_id", n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """Per config: mean + median over units (two-stage aggregated) and
    cluster-bootstrap CI for the mean.

    Point estimate : mean of the subject-level means.
    Median         : median of the same subject-level means (descriptive).
    Uncertainty    : percentile CI from resampling `unit` with replacement.
    `case_wise` is accepted for backwards compatibility and ignored.
    """
    metrics = available_metrics(df, metrics)
    level = unit_level(df, metrics, unit=unit, case=case)
    units = sorted(df[unit].unique())
    w = bootstrap_weights(units, n_boot, seed)
    one = np.ones((1, len(units)))

    order = _config_order(df)
    median = level.groupby(level="config", observed=True)[metrics].median().reindex(order)

    per_cfg = df.groupby("config", observed=True)
    rows, index = [], []
    for cfg in order:
        v, m = _unit_matrix(level, cfg, units, metrics)
        point = _unit_means(v, m, one)[0]
        boot = _unit_means(v, m, w)
        lo, hi = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
        se = np.nanstd(boot, axis=0, ddof=1)
        rows.append(np.concatenate([point, lo, hi, se,
                                    median.loc[cfg].to_numpy(float)]))
        index.append(cfg)

    cols = (metrics + [f"{k}_lo" for k in metrics]
            + [f"{k}_hi" for k in metrics] + [f"{k}_se" for k in metrics]
            + [f"{k}_median" for k in metrics])
    out = pd.DataFrame(rows, index=pd.Index(index, name="config"), columns=cols)
    out.insert(0, "n_units", per_cfg[unit].nunique().reindex(index).to_numpy())
    out.insert(1, "n_cases", per_cfg[case].nunique().reindex(index).to_numpy())
    out.insert(2, "n_cs", per_cfg.size().reindex(index).to_numpy())
    return out

def paired_bootstrap(df, ref, metrics=None, case_wise=None, unit="subject_id",
                     case="case_id", n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """Cluster-bootstrap CI for the paired difference (reference - other) of the
    subject-level means, using the *same* resamples as `bootstrap_summary`."""
    metrics = available_metrics(df, metrics)
    level = unit_level(df, metrics, unit=unit, case=case)
    units = sorted(df[unit].unique())
    w = bootstrap_weights(units, n_boot, seed)
    one = np.ones((1, len(units)))

    vr, mr = _unit_matrix(level, ref, units, metrics)
    boot_ref = _unit_means(vr, mr, w)
    point_ref = _unit_means(vr, mr, one)[0]

    rows = []
    for cfg in _config_order(df):
        if cfg == ref:
            continue
        v, m = _unit_matrix(level, cfg, units, metrics)
        d_boot = boot_ref - _unit_means(v, m, w)
        d_point = point_ref - _unit_means(v, m, one)[0]
        lo, hi = np.nanpercentile(d_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
        for j, k in enumerate(metrics):
            rows.append({"config": cfg, "metric": k, "boot_diff": d_point[j],
                         "boot_lo": lo[j], "boot_hi": hi[j],
                         "boot_excludes_zero": bool(lo[j] > 0 or hi[j] < 0)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# descriptive spread (cross-section level, NOT used for inference)
# --------------------------------------------------------------------------- #
def _quant(g, metrics):
    q25, q50, q75 = g.quantile(0.25), g.median(), g.quantile(0.75)
    p05, p95 = g.quantile(0.05), g.quantile(0.95)
    return pd.concat([q50.add_suffix("_median"), q25.add_suffix("_q25"),
                      q75.add_suffix("_q75"),
                      (q75 - q25).add_suffix("_iqr"),
                      p05.add_suffix("_p05"), p95.add_suffix("_p95")], axis=1)


def descriptive_summary(df, metrics=None, case_wise=None, unit="subject_id",
                        case="case_id"):
    """Median / IQR / 5-95 pct of the subject-level means (descriptive only)."""
    metrics = available_metrics(df, metrics)
    level = unit_level(df, metrics, unit=unit, case=case)
    return _quant(level.groupby("config", observed=True)[metrics], metrics)


def summarise(case: pd.DataFrame, order=None, quantiles: bool = True,
              metrics=None) -> pd.DataFrame:
    """Per-config mean / sd / 95 % CI (+ median/quartiles) and n_cases."""
    metrics = [k for k in (METRICS if metrics is None else metrics) if k in case.columns]
    g = case.groupby("config", observed=True)[metrics]
    mean, sd, n = g.mean(), g.std(ddof=1), g.size()
    lo, hi = t.interval(1 - ALPHA, df=(n.to_numpy() - 1)[:, None],
                        loc=mean.to_numpy(), scale=g.sem().to_numpy())
    parts = [mean, sd.add_suffix("_sd"),
             pd.DataFrame(lo, mean.index, [f"{k}_lo" for k in metrics]),
             pd.DataFrame(hi, mean.index, [f"{k}_hi" for k in metrics])]
    if quantiles:
        q25, q75 = g.quantile(0.25), g.quantile(0.75)
        parts += [g.median().add_suffix("_median"),
                  q25.add_suffix("_q25"), q75.add_suffix("_q75"),
                  pd.DataFrame(q75.to_numpy() - q25.to_numpy(), mean.index,
                               [f"{k}_iqr" for k in metrics])]
    out = pd.concat(parts, axis=1)
    out.insert(0, "n_cases", n)
    return out if order is None else out.loc[list(order)]


# --------------------------------------------------------------------------- #
# paired tests
# --------------------------------------------------------------------------- #
def hodges_lehmann(d):
    """Median of Walsh averages + signed-rank-compatible CI."""
    d = np.asarray(d, float)
    n = d.size
    walsh = np.sort(np.add.outer(d, d)[np.triu_indices(n)] / 2.0)
    k = int(np.floor(n * (n + 1) / 4
                     - norm.ppf(1 - ALPHA / 2) * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = int(np.clip(k, 0, walsh.size - 1))
    return float(np.median(walsh)), float(walsh[k]), float(walsh[-1 - k])


def paired_tests(case, ref, metrics=None, deltas=None) -> pd.DataFrame:
    """Paired Wilcoxon (reference vs each other config) per metric, Holm-corrected.

    Metrics that are entirely NaN (e.g. no wall segmentation) are skipped instead
    of removing every case via `dropna`.
    """
    metrics = available_metrics(case, metrics)
    base = case.xs(ref, level="config")[metrics]
    rows = []
    for cfg, g in case.groupby("config", observed=True):
        if cfg == ref:
            continue
        d = (base - g.droplevel("config")[metrics]).dropna(how="all")   # reference - other
        for k in metrics:
            x = d[k].dropna().to_numpy(float)
            n = x.size
            if n < 2:
                print(f"  [utils] skipping {cfg}/{k}: n={n}")
                continue
            ci_lo, ci_hi = t.interval(1 - ALPHA, n - 1,
                                      loc=x.mean(), scale=x.std(ddof=1) / np.sqrt(n))
            nz = int(np.count_nonzero(x))
            hl, hl_lo, hl_hi = hodges_lehmann(x)
            rows.append({
                "config": cfg, "metric": k, "n": n, "n_nonzero": nz, "n_zero": n - nz,
                "mean_diff": x.mean(), "median_diff": float(np.median(x)),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "hl_diff": hl, "hl_lo": hl_lo, "hl_hi": hl_hi,
                "favours": "reference" if (hl < 0) == LOWER_IS_BETTER[k] else "other",
                "hl_excludes_zero": (hl_lo > 0) or (hl_hi < 0),
                "p_raw": wilcoxon(x, zero_method=ZERO_METHOD).pvalue if nz else 1.0,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = multipletests(out["p_raw"], method="holm")[1]
    out["sig"] = out["p_holm"] < ALPHA
    out["direction"] = np.where(~out["sig"], "",
                                np.where(out["favours"] == "reference", "worse", "better"))
    if deltas is not None and not deltas.empty:
        out = out.merge(deltas, on=["config", "metric"], how="left")
    return out


def significance_marks(tests, configs=None) -> dict:
    """(config, metric) -> LaTeX marker."""
    d = tests if configs is None else tests[tests["config"].isin(list(configs))]
    return {(r.config, r.metric): MARK_SIG[r.direction] for r in d.itertuples()}


# --------------------------------------------------------------------------- #
# LaTeX formatting
# --------------------------------------------------------------------------- #
def best_values(d: pd.DataFrame, metrics=TABLE_METRICS, suffix: str = "") -> dict:
    """Best value per metric; `suffix` selects a statistic column, e.g. "_median"."""
    return {k: (d[k + suffix].min() if LOWER_IS_BETTER[k] else d[k + suffix].max())
            for k in metrics}

def fmt(value, best=np.nan, mark="", bold=BOLDFMT, dec=DEC) -> str:
    if not np.isfinite(value):
        return "--"
    s = f"{value:.{dec}f}"
    return (bold.format(s) if np.isfinite(best) and np.isclose(value, best) else s) + mark


def fmt_mean_ci(mean, lo, hi, best=np.nan, mark="", dec=DEC, two_line=True) -> str:
    """mean with 95 % CI underneath; \\multicolumn shields it from siunitx."""
    if not np.isfinite(mean):
        return r"\multicolumn{1}{c}{--}"
    head = fmt(mean, best, mark, dec=dec)
    ci = rf"({lo:.{dec}f}--{hi:.{dec}f})"
    body = rf"\makecell{{{head}\\[-2pt]\tiny {ci}}}" if two_line else f"{head}~{ci}"
    return rf"\multicolumn{{1}}{{c}}{{{body}}}"


def fmt_delta_ci(d, lo, hi, mark="", dec=DEC) -> str:
    if not np.isfinite(d):
        return r"\multicolumn{1}{c}{--}"
    body = (rf"\makecell{{{d:+.{dec}f}{mark}\\[-2pt]"
            rf"\scriptsize ({lo:+.{dec}f}--{hi:+.{dec}f})}}")
    return rf"\multicolumn{{1}}{{c}}{{{body}}}"


def metric_header(metrics, n_factor_cols, factor_names, extra_top="", extra_bottom="",
                  score_label="Score"):
    """Two-row grouped header built from METRIC_HEADER."""
    top, bottom, i = [""] * n_factor_cols, list(factor_names), 0
    while i < len(metrics):
        grp, sub = METRIC_HEADER.get(metrics[i], (metrics[i], ""))
        if sub == "":
            top.append(grp)
            bottom.append("")
            i += 1
            continue
        span = [metrics[i]]
        while (i + len(span) < len(metrics)
               and METRIC_HEADER.get(metrics[i + len(span)], ("", ""))[0] == grp):
            span.append(metrics[i + len(span)])
        top.append(rf"\multicolumn{{{len(span)}}}{{c}}{{{grp}}}")
        top += [""] * (len(span) - 1)
        bottom += [METRIC_HEADER[m][1] for m in span]
        i += len(span)
    top.append("")
    bottom.append(score_label)
    return [" & ".join(top) + extra_top + r" \\",
            " & ".join(bottom) + extra_bottom + r" \\ \hline"]
