#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# aggregation2.py
import pandas as pd
from means import (
    trimmed_mean_prob,
    geometric_mean_prob,
    geometric_mean_odds,
)
from carry_forward import carry_forward_snapshots

# Columns we renormalize per (question, day)
_PROB_COLS = [
    "mean_probability",
    "median_probability",
    "trimmed_mean_probability",
    "geometric_mean_probability",
    "geometric_mean_odds",
]


def aggregate_across_forecasters(snapshots):
    """
    Pool snapshots across forecasters into the five aggregation methods,
    keep resolved_probability from snapshots, and renormalize per (question, day).
    """
    if snapshots.empty:
        cols = [
            "discover question id", "day", "answer_sort_order",
            "mean_probability", "median_probability", "trimmed_mean_probability",
            "geometric_mean_probability", "geometric_mean_odds",
            "n_forecasters", "resolved_probability",
        ]
        return pd.DataFrame(columns=cols)

    agg = (
        snapshots
        .groupby(["discover question id", "day", "answer sort order"], as_index=False)
        .agg(
            mean_probability=("prob", "mean"),
            median_probability=("prob", "median"),
            trimmed_mean_probability=("prob", trimmed_mean_prob),
            geometric_mean_probability=("prob", geometric_mean_prob),
            geometric_mean_odds=("prob", geometric_mean_odds),
            n_forecasters=("membership guid", "nunique"),
            resolved_probability=("resolved_probability", "first"),  # constant per (qid, answer)
        )
        .rename(columns={"answer sort order": "answer_sort_order"})
        .sort_values(["discover question id", "day", "answer_sort_order"], kind="stable")
        .reset_index(drop=True)
    )

    # Vectorized renormalization within (qid, day), skip if only one bucket
    sizes = agg.groupby(["discover question id", "day"])["answer_sort_order"].transform("size")
    multi_bucket = sizes > 1

    for col in _PROB_COLS:
        totals = agg.groupby(["discover question id", "day"])[col].transform("sum")
        mask = (multi_bucket) & (totals > 0)
        agg.loc[mask, col] = agg.loc[mask, col] / totals[mask]

    # Downcast to save memory
    for col in _PROB_COLS:
        agg[col] = agg[col].astype("float32")
    if "resolved_probability" in agg:
        agg["resolved_probability"] = agg["resolved_probability"].astype("float32")

    return agg


def build_carry_forward_aggregates(
    forecaster_day,
    max_staleness_days=None,
):
    """
    Carry-forward snapshots -> pooled aggregates.
    """
    # Use the shared engine; pass through truth so we don't need extra merges.
    snapshots = carry_forward_snapshots(
        forecaster_day,
        max_staleness_days=max_staleness_days,
        per_answer_cols=("resolved_probability",),   # keep truth on snapshots
        per_question_cols=(),                         # aggregator doesn’t need question-level cols
        downcast=True,
    )
    return aggregate_across_forecasters(snapshots)
