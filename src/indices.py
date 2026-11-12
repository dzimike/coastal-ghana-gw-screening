"""
src/indices.py
==============
Composite index construction for the Ghana groundwater-subsidence project.

Implements the Urban Groundwater Pressure Index (UGPI) and the Coastal
Groundwater-Subsidence Risk Index (CGSRI) described in CLAUDE.md §9–§10.

All indicators are normalised to [0, 1] using min-max normalisation before
weighting. Higher normalised values always indicate greater pressure or risk.
"Inverse" indicators (where a higher raw value means less risk, e.g. piped-
water access) must be inverted before or after normalisation as appropriate.

Public API
----------
    minmax_normalize(series, feature_range, clip)
    invert_indicator(series)
    weighted_sum_index(df_indicators, weights, normalise_first)
    classify_quantiles(series, n_classes, labels)
    classify_natural_breaks(series, n_classes, labels)
    build_ugpi(df, weights, id_col, inverse_indicators)
    build_cgsri(df, weights, id_col, inverse_indicators, ...)
    run_sensitivity_analysis(df_indicators, weight_scenarios, classify_fn, id_col)
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Min-max normalisation
# ---------------------------------------------------------------------------
def minmax_normalize(
    series: pd.Series,
    feature_range: tuple[float, float] = (0.0, 1.0),
    clip: bool = True,
) -> pd.Series:
    """
    Normalise a numeric series to [0, 1] (or a custom range) using min-max
    scaling.

        X_norm = (X − X_min) / (X_max − X_min)

    Parameters
    ----------
    series : pd.Series
        Input indicator values. NaN values are preserved.
    feature_range : tuple of float
        Target output range, default (0.0, 1.0).
    clip : bool
        Clip output values to ``feature_range`` (default True).

    Returns
    -------
    pd.Series
        Normalised values.

    Raises
    ------
    UserWarning
        If all non-NaN values are identical (zero-variance indicator).
    """
    a, b = feature_range
    x_min = series.min(skipna=True)
    x_max = series.max(skipna=True)

    if np.isclose(x_min, x_max):
        warnings.warn(
            f"All values in series '{series.name}' are identical ({x_min}). "
            "Min-max normalisation is undefined. Returning mid-point for all non-NaN.",
            UserWarning,
            stacklevel=2,
        )
        result = series.copy().astype(float)
        result[~series.isna()] = (a + b) / 2.0
        return result

    norm = (series - x_min) / (x_max - x_min) * (b - a) + a
    if clip:
        norm = norm.clip(lower=a, upper=b)
    norm.name = series.name
    return norm


# ---------------------------------------------------------------------------
# 2. Invert an indicator
# ---------------------------------------------------------------------------
def invert_indicator(series: pd.Series) -> pd.Series:
    """
    Invert a normalised indicator so that higher values indicate greater risk.

        X_inv = 1 − X_norm   (assumes input is in [0, 1])

    Use for "positive" indicators where a higher raw value means lower risk,
    for example piped-water access (more access = less groundwater pressure).

    Parameters
    ----------
    series : pd.Series
        Indicator values, expected to be in [0, 1].

    Returns
    -------
    pd.Series
        Inverted values in [0, 1].
    """
    if series.dropna().lt(0).any() or series.dropna().gt(1).any():
        warnings.warn(
            f"Series '{series.name}' contains values outside [0, 1]. "
            "Ensure normalisation has been applied before inverting.",
            UserWarning,
            stacklevel=2,
        )
    result = 1.0 - series
    result.name = (
        f"{series.name}_inverted" if series.name else "inverted"
    )
    return result


# ---------------------------------------------------------------------------
# 3. Weighted sum index
# ---------------------------------------------------------------------------
def weighted_sum_index(
    df_indicators: pd.DataFrame,
    weights: dict[str, float],
    normalise_first: bool = True,
) -> pd.Series:
    """
    Compute a weighted sum composite index from a DataFrame of indicators.

        Index = Σ(w_i × X_i_norm)

    Parameters
    ----------
    df_indicators : pd.DataFrame
        Each column is one indicator. Rows are spatial units.
    weights : dict
        Column name → weight. Weights should sum to 1.0.
    normalise_first : bool
        If True (default), apply ``minmax_normalize`` to each column
        before weighting. Set False if columns are already normalised.

    Returns
    -------
    pd.Series
        Composite index values, one per row, named ``"composite_index"``.

    Raises
    ------
    KeyError
        If any key in ``weights`` is absent from ``df_indicators.columns``.
    """
    missing = [k for k in weights if k not in df_indicators.columns]
    if missing:
        raise KeyError(
            f"Weight keys not found in DataFrame: {missing}. "
            f"Available columns: {list(df_indicators.columns)}"
        )

    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        warnings.warn(
            f"Weights sum to {weight_sum:.6f} (expected 1.0). "
            "Index will be proportionally scaled.",
            UserWarning,
            stacklevel=2,
        )

    index = pd.Series(0.0, index=df_indicators.index, name="composite_index")
    for col, w in weights.items():
        col_vals = df_indicators[col].copy()
        if normalise_first:
            col_vals = minmax_normalize(col_vals)
        index = index + w * col_vals

    log.info(
        "Weighted sum index: %d units, %d complete (no NaN).",
        len(index), int(index.notna().sum()),
    )
    return index


# ---------------------------------------------------------------------------
# 4. Classify by quantiles
# ---------------------------------------------------------------------------
def classify_quantiles(
    series: pd.Series,
    n_classes: int = 5,
    labels: list[str] | None = None,
) -> pd.Series:
    """
    Classify a continuous index into equal-frequency (quantile) classes.

    Parameters
    ----------
    series : pd.Series
        Continuous index values.
    n_classes : int
        Number of classes (default 5).
    labels : list of str or None
        Class labels in ascending order (must have ``n_classes`` elements).
        Default: ["Very low", "Low", "Moderate", "High", "Very high"] for n=5.

    Returns
    -------
    pd.Series
        String class labels, same index as ``series``.
    """
    if labels is None:
        labels = _default_labels(n_classes)
    if len(labels) != n_classes:
        raise ValueError(
            f"len(labels)={len(labels)} must equal n_classes={n_classes}."
        )

    result = pd.qcut(
        series, q=n_classes, labels=labels, duplicates="drop"
    ).astype(str)
    result.name = f"{series.name}_class" if series.name else "class"
    _log_class_counts(result, "quantile")
    return result


# ---------------------------------------------------------------------------
# 5. Classify by natural breaks (Fisher-Jenks)
# ---------------------------------------------------------------------------
def classify_natural_breaks(
    series: pd.Series,
    n_classes: int = 5,
    labels: list[str] | None = None,
) -> pd.Series:
    """
    Classify a continuous index using Fisher-Jenks natural breaks.

    Natural breaks minimise within-class variance and are appropriate when
    the distribution is multimodal or has clear clusters. Falls back to
    quantile classification if ``mapclassify`` is not installed.

    Parameters
    ----------
    series : pd.Series
        Continuous index values.
    n_classes : int
        Number of classes (default 5).
    labels : list of str or None
        Class labels in ascending order.

    Returns
    -------
    pd.Series
        String class labels.

    Notes
    -----
    Requires: pip install mapclassify
    """
    if labels is None:
        labels = _default_labels(n_classes)
    if len(labels) != n_classes:
        raise ValueError(
            f"len(labels)={len(labels)} must equal n_classes={n_classes}."
        )

    try:
        import mapclassify  # type: ignore
    except ImportError:
        warnings.warn(
            "mapclassify not installed; falling back to quantile classification. "
            "Install with: pip install mapclassify",
            ImportWarning,
            stacklevel=2,
        )
        return classify_quantiles(series, n_classes, labels)

    clean = series.dropna()
    nb = mapclassify.NaturalBreaks(clean.values, k=n_classes)
    bins = [-np.inf] + list(nb.bins)

    result = series.copy().astype(object)
    for i, label in enumerate(labels):
        lo, hi = bins[i], bins[i + 1]
        mask = (series > lo) & (series <= hi)
        result[mask] = label
    result[series.isna()] = np.nan
    result.name = f"{series.name}_class" if series.name else "class"
    _log_class_counts(result, "natural breaks")
    return result


# ---------------------------------------------------------------------------
# 6. Build Urban Groundwater Pressure Index (UGPI)
# ---------------------------------------------------------------------------
def build_ugpi(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    id_col: str = "cell_id",
    inverse_indicators: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the Urban Groundwater Pressure Index (UGPI).

        UGPI = Σ(w_i × X_i_norm)

    Expected indicator columns (names must match ``weights`` keys):
        building_density      — building footprints per km²
        population_density    — persons per km²
        builtup_growth_rate   — % annual change in built-up area
        borehole_density      — boreholes per km²
        industrial_intensity  — industrial land-use fraction
        piped_water_access    — fraction with piped water (INVERSE: more = less pressure)
        drought_frequency     — fraction of months classified as drought

    Parameters
    ----------
    df : pd.DataFrame
        One row per spatial unit with indicator columns and ``id_col``.
    weights : dict or None
        Column → weight mapping summing to 1.0.
        Default: equal weights from ``config/thresholds.yaml``.
    id_col : str
        Column identifying each spatial unit.
    inverse_indicators : list of str or None
        Indicator columns to invert after normalisation.
        Default: ["piped_water_access"].

    Returns
    -------
    pd.DataFrame
        Original ``df`` plus columns:
            ``ugpi``        — continuous pressure score [0, 1]
            ``ugpi_class``  — quantile class label
    """
    from src.config import thresholds as cfg

    if weights is None:
        weights = cfg["ugpi"]["weights"]
    if inverse_indicators is None:
        inverse_indicators = ["piped_water_access"]

    log.info("Building UGPI for %d spatial units ...", len(df))
    df_work = df.set_index(id_col).copy()

    df_norm = pd.DataFrame(index=df_work.index)
    for col in weights:
        if col not in df_work.columns:
            log.warning("Indicator '%s' not found; filling with NaN.", col)
            df_norm[col] = np.nan
            continue
        normed = minmax_normalize(df_work[col])
        if col in inverse_indicators:
            normed = invert_indicator(normed)
        df_norm[col] = normed

    ugpi = weighted_sum_index(df_norm, weights, normalise_first=False)
    ugpi.name = "ugpi"

    df_out = df.copy()
    df_out["ugpi"] = ugpi.values
    df_out["ugpi_class"] = classify_quantiles(ugpi).values

    log.info(
        "UGPI built. Score range: %.3f – %.3f.",
        df_out["ugpi"].min(), df_out["ugpi"].max(),
    )
    return df_out


# ---------------------------------------------------------------------------
# 7. Build Coastal Groundwater-Subsidence Risk Index (CGSRI)
# ---------------------------------------------------------------------------
def build_cgsri(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    id_col: str = "cell_id",
    inverse_indicators: list[str] | None = None,
    classification_method: str = "natural_breaks",
    n_classes: int = 5,
    class_labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the Coastal Groundwater-Subsidence Risk Index (CGSRI).

        CGSRI = α·GW_Hazard + β·Subsidence_Hazard + γ·Urban_Pressure
                + δ·Climate_Stress + ε·Exposure + ζ·Vulnerability

    Expected indicator columns (names must match ``weights`` keys):
        gw_hazard            — composite groundwater hazard score
        subsidence_hazard    — composite subsidence hazard score
        urban_pressure       — UGPI or normalised urban pressure
        climate_stress       — normalised rainfall deficit / PDSI stress
        exposure             — normalised exposure score (population, infrastructure)
        vulnerability        — normalised vulnerability score

    Parameters
    ----------
    df : pd.DataFrame
        One row per spatial unit with composite indicator columns and ``id_col``.
    weights : dict or None
        Column → weight mapping. Default from ``config/thresholds.yaml``.
    id_col : str
        Column identifying each spatial unit.
    inverse_indicators : list of str or None
        Indicator columns to invert after normalisation (if any).
    classification_method : str
        "natural_breaks" (default) or "quantiles".
    n_classes : int
        Number of risk classes (default 5).
    class_labels : list of str or None
        Risk class labels in ascending order of risk.

    Returns
    -------
    pd.DataFrame
        Original ``df`` plus columns:
            ``cgsri``       — continuous risk score [0, 1]
            ``cgsri_class`` — classified risk label
    """
    from src.config import thresholds as cfg

    if weights is None:
        weights = cfg["risk_index"]["weights"]
    if class_labels is None:
        class_labels = cfg["risk_index"]["class_labels"]
    if not classification_method:
        classification_method = cfg["risk_index"]["classification_method"]
    if inverse_indicators is None:
        inverse_indicators = []

    log.info("Building CGSRI for %d spatial units ...", len(df))
    df_work = df.set_index(id_col).copy()

    df_norm = pd.DataFrame(index=df_work.index)
    for col in weights:
        if col not in df_work.columns:
            log.warning("Indicator '%s' not found; filling with NaN.", col)
            df_norm[col] = np.nan
            continue
        normed = minmax_normalize(df_work[col])
        if col in inverse_indicators:
            normed = invert_indicator(normed)
        df_norm[col] = normed

    cgsri = weighted_sum_index(df_norm, weights, normalise_first=False)
    cgsri.name = "cgsri"

    classify_fn = (
        classify_natural_breaks
        if classification_method == "natural_breaks"
        else classify_quantiles
    )
    cgsri_class = classify_fn(cgsri, n_classes=n_classes, labels=class_labels)

    df_out = df.copy()
    df_out["cgsri"] = cgsri.values
    df_out["cgsri_class"] = cgsri_class.values

    log.info(
        "CGSRI built. Score range: %.3f – %.3f. Classification: %s.",
        df_out["cgsri"].min(), df_out["cgsri"].max(), classification_method,
    )
    return df_out


# ---------------------------------------------------------------------------
# 8. Sensitivity analysis
# ---------------------------------------------------------------------------
def run_sensitivity_analysis(
    df_indicators: pd.DataFrame,
    weight_scenarios: dict[str, dict[str, float]],
    classify_fn: Callable[[pd.Series], pd.Series] | None = None,
    id_col: str = "cell_id",
) -> pd.DataFrame:
    """
    Run a multi-scenario weight sensitivity analysis on a composite index.

    Computes the composite index under each weight scenario, classifies it,
    and returns results in long format. Pairwise Spearman rank correlations
    between scenario scores are logged — high correlations (> 0.90) indicate
    that rankings are robust to weight choice.

    Parameters
    ----------
    df_indicators : pd.DataFrame
        Normalised indicator columns plus ``id_col``.
    weight_scenarios : dict
        Scenario name → weight dict mapping. Example::

            {
                "equal":  {"gw_hazard": 0.17, "subsidence_hazard": 0.17, ...},
                "expert": {"gw_hazard": 0.30, "subsidence_hazard": 0.25, ...},
            }

    classify_fn : callable or None
        Function ``(pd.Series) → pd.Series`` for classifying the index.
        Default: ``classify_quantiles`` with 5 classes.
    id_col : str
        Column identifying each spatial unit.

    Returns
    -------
    pd.DataFrame
        Long-format with columns: ``scenario``, ``id_col``, ``score``, ``risk_class``.
    """
    if classify_fn is None:
        classify_fn = lambda s: classify_quantiles(s, n_classes=5)

    df_work = df_indicators.set_index(id_col).copy()
    records = []

    for scenario_name, weights in weight_scenarios.items():
        log.info("Sensitivity analysis: running scenario '%s' ...", scenario_name)
        score = weighted_sum_index(df_work, weights, normalise_first=True)
        classes = classify_fn(score)
        for unit_id, s, c in zip(df_work.index, score, classes):
            records.append({
                id_col:       unit_id,
                "scenario":   scenario_name,
                "score":      s,
                "risk_class": c,
            })

    df_out = pd.DataFrame(records)

    # Log pairwise Spearman rank correlations
    scenario_names = list(weight_scenarios.keys())
    if len(scenario_names) > 1:
        pivot = df_out.pivot(index=id_col, columns="scenario", values="score")
        corr = pivot.rank().corr(method="spearman")
        log.info(
            "Pairwise Spearman rank correlations between weight scenarios:\n%s",
            corr.to_string(),
        )

    return df_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _default_labels(n: int) -> list[str]:
    """Return default ascending risk class labels for n classes."""
    defaults = {
        2: ["Low", "High"],
        3: ["Low", "Moderate", "High"],
        4: ["Low", "Moderate", "High", "Very high"],
        5: ["Very low", "Low", "Moderate", "High", "Very high"],
    }
    return defaults.get(n, [f"Class {i + 1}" for i in range(n)])


def _log_class_counts(series: pd.Series, method: str) -> None:
    counts = series.value_counts(dropna=False)
    log.info("Classification (%s) class counts:\n%s", method, counts.to_string())
