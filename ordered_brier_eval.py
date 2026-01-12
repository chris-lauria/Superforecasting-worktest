#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


def ordered_brier_from_distribution(probs: np.ndarray,
                                    outcome: np.ndarray) -> float:
    """
    Ordered Brier score (Jose, Nau, Winkler 2009) for an ordered categorical forecast.

    Assumes hard resolution: exactly one category is the true outcome.
    """

    probs = np.asarray(probs, dtype=float)
    outcome = np.asarray(outcome, dtype=float)

    if probs.ndim != 1 or outcome.ndim != 1:
        raise ValueError("probs and outcome must be 1-D arrays.")
    if probs.shape[0] != outcome.shape[0]:
        raise ValueError("probs and outcome must have same length.")

    K = probs.shape[0]

    # If only one bucket was provided, construct the implicit complement.
    # Example input:
    #   probs   = [0.6]
    #   outcome = [1]
    # becomes:
    #   probs   = [0.6, 0.4]
    #   outcome = [1.0, 0.0]
    if K == 1:
        p = probs[0]
        y = outcome[0]
        probs = np.array([p, 1.0 - p], dtype=float)
        outcome = np.array([y, 1.0 - y], dtype=float)
        K = 2

    # ---- STRICT CHECK: outcome must be exactly one-hot ----
    # Requirements:
    # - All entries are either 0 or 1 (no other values)
    # - Exactly one entry is 1
    # - Everything else is 0
    unique_vals = np.unique(outcome)

    # Rule 1: outcome can ONLY contain 0 and/or 1
    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        raise ValueError(
            "Outcome must be one-hot: entries must be 0 or 1 only."
        )

    # Rule 2: exactly one 1
    if np.sum(outcome == 1.0) != 1:
        raise ValueError(
            "Outcome must be one-hot: exactly one category must be 1."
        )

    # We do NOT renormalize outcome (truth is exact), but we DO
    # gently renormalize forecast probs in case they don't sum to 1 exactly.
    psum = probs.sum()
    if psum > 0:
        probs = probs / psum

    # cumulative forecast and cumulative truth
    cum_forecast = np.cumsum(probs)   # length K
    cum_truth   = np.cumsum(outcome)  # length K

    # For each boundary j = 0 .. K-2, make a binary split:
    #   left side = categories <= j
    #   right side = categories > j
    # predicted "yes" prob at split j = cum_forecast[j]
    # actual "yes" outcome at split j = cum_truth[j]
    #
    # Binary Brier score for that split is:
    #   (p_yes - y_yes)^2 + (p_no - y_no)^2
    # which simplifies to:
    #   2 * (p_yes - y_yes)^2
    diffsq = (cum_forecast[:-1] - cum_truth[:-1]) ** 2
    brier_per_split = 2.0 * diffsq

    # Ordered Brier score = average across splits
    score = brier_per_split.mean()

    return float(score)



def _score_one_day(group: pd.DataFrame):
    """
    Helper: given a subset of agg for ONE (discover question id, day),
    compute ordered Brier score for each aggregation method.

    group is expected to have columns:
        "answer_sort_order"
        "mean_probability"
        "median_probability"
        "trimmed_mean_probability"
        "geometric_mean_probability"
        "geometric_mean_odds"
        "resolved_probability"

    Returns a dict with that day's scores.
    """

    # sort answers in their ordinal order
    g = group.sort_values("answer_sort_order")

    # ground truth vector in order
    outcome = g["resolved_probability"].to_numpy(dtype=float)

    # build forecast vectors for each aggregation method
    mean_p = g["mean_probability"].to_numpy(dtype=float)
    median_p = g["median_probability"].to_numpy(dtype=float)
    trimmed_p = g["trimmed_mean_probability"].to_numpy(dtype=float)
    geom_p = g["geometric_mean_probability"].to_numpy(dtype=float)
    geom_odds_p = g["geometric_mean_odds"].to_numpy(dtype=float)

    return {
        "obs_mean": ordered_brier_from_distribution(mean_p, outcome),
        "obs_median": ordered_brier_from_distribution(median_p, outcome),
        "obs_trimmed": ordered_brier_from_distribution(trimmed_p, outcome),
        "obs_geom_prob": ordered_brier_from_distribution(geom_p, outcome),
        "obs_geom_odds": ordered_brier_from_distribution(geom_odds_p, outcome),
        # carry some metadata for grouping later
        "discover question id": g["discover question id"].iloc[0],
        "day": g["day"].iloc[0],
    }


def compute_ordered_brier_for_aggregates(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point.

    Input
    -----
    agg_df : DataFrame like the one you produced in Version B, with columns:
        "discover question id"
        "day"
        "answer_sort_order"
        "mean_probability"
        "median_probability"
        "trimmed_mean_probability"
        "geometric_mean_probability"
        "geometric_mean_odds"
        "resolved_probability"

    Output
    ------
    day_scores : DataFrame with one row per (discover question id, day):
        "discover question id"
        "day"
        "obs_mean"
        "obs_median"
        "obs_trimmed"
        "obs_geom_prob"
        "obs_geom_odds"

    You can then:
      - average these columns over time to get an overall score,
      - or average within each question.
    """

    results = []
    for (qid, day), sub in agg_df.groupby(["discover question id", "day"]):
        res = _score_one_day(sub)
        results.append(res)

    day_scores = pd.DataFrame(results).sort_values(
        ["discover question id", "day"]
    ).reset_index(drop=True)

    return day_scores


def summarize_ordered_brier(day_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Optional convenience: take the per-day scores from
    compute_ordered_brier_for_aggregates and summarize them.

    Returns two levels of summary:
    1. per-question average score across all its days
    2. global average across all questions/days (row 'ALL')

    Output columns:
        "obs_mean", "obs_median", "obs_trimmed", "obs_geom_prob", "obs_geom_odds"
    Lower = better.
    """

    # per-question average
    by_q = (
        day_scores
        .groupby("discover question id", as_index=False)[
            ["obs_mean", "obs_median", "obs_trimmed",
             "obs_geom_prob", "obs_geom_odds"]
        ]
        .mean()
    )

    # global overall average
    overall = (
        day_scores[
            ["obs_mean", "obs_median", "obs_trimmed",
             "obs_geom_prob", "obs_geom_odds"]
        ]
        .mean()
        .to_frame()
        .T
    )
    overall.insert(0, "discover question id", "ALL")

    summary = pd.concat([by_q, overall], ignore_index=True)
    return summary



# print(ordered_brier_from_distribution( [0,0.5,0.25,0.25], [0,1,0,0]))



# print(ordered_brier_from_distribution( [0.6], [1]))
























