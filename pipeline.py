#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: 
"""

# pipelines.py
import math

import pandas as pd

from carry_forward import carry_forward_snapshots
from individual_scoring import score_individuals
from aggregation import aggregate_across_forecasters
from ordered_brier_eval import compute_ordered_brier_for_aggregates, summarize_ordered_brier
from build_table import build_paper_table


# ---------------------------
# Shared helpers
# ---------------------------



# This function convert a compact per-forecaster/per-day table (`forecaster_day`) into
#   a full set of per-day “snapshots” by carrying each forecaster’s last
#   submitted probabilities forward to future days as is canonical in forecast
#   competitions

def _build_snapshots(forecaster_day, max_staleness_days):
    """Carry-forward snapshots with truth; used by both pipelines."""
    return carry_forward_snapshots(
        forecaster_day,
        max_staleness_days=max_staleness_days,
        per_answer_cols=("resolved_probability",),  
        per_question_cols=(),                       
        downcast=True,
    )


#   This function takes per-day snapshots, aggregate across
#   forecasters, computes Ordered Brier scores, summarizes, and build a
#   compact table view.

def _run_aggregate_from_snapshots(snapshots, *, label, agg_out_base="question_day_means"):
    """Core: snapshots -> aggregates -> scores -> files -> paper table."""
    # Aggregate across forecasters
    agg = aggregate_across_forecasters(snapshots)

    # Sanity: (qid, day) sums ~ 1
    sums = agg.groupby(["discover question id", "day"], observed=True)["mean_probability"].sum()
    off = (sums - 1.0).abs()
    n_off = int((off > 3e-2).sum())
    if n_off:
        print(f"[{label}] Note: {n_off} question-day groups do not sum exactly to 1 (tol=3e-2).")

    # Save aggregated forecasts  if you want to debug
    
    # agg_out_path = f"{agg_out_base}_{label}.csv"
    # agg.to_csv(agg_out_path, index=False)
    # print(f"[{label}] Wrote aggregated means to: {Path(agg_out_path).resolve()}")
  
    

    # Ordered Brier per day (aggregate level) + summaries
    day_scores = compute_ordered_brier_for_aggregates(agg)
    summary = summarize_ordered_brier(day_scores)
    paper_table = build_paper_table(day_scores)

    # Dumps (write directly; avoid building large CSV strings in memory)
    # day_scores.to_csv(f"scores_per_day_{label}.csv", index=False)
    # summary.to_csv(f"scores_summary_{label}.csv", index=False)
    # paper_table.to_csv(f"paper_table_{label}.csv", index=False)

    # print(f"\n[{label}] Per-day ordered Brier scores (first few rows):")
    # print(day_scores.head())
    # print(f"\n[{label}] Summary (per question + overall):")
    # print(summary)
    print(f"\n[{label}] Compact table:")
    print(paper_table)

    return {
        "agg": agg,
        "day_scores": day_scores,
        "summary": summary,
        "paper_table": paper_table,
    }


# ---------------------------
# Public runners
# ---------------------------

#   Convenience wrapper for the “plain” pipeline (no trimming). Builds
#   carry-forward snapshots with a chosen staleness window, then runs the
#   core aggregation + scoring pipeline.

def run_aggregate_variant(forecaster_day, *, label, max_staleness_days, agg_out_base="question_day_means"):
    """Plain variant (e.g., 'nostale', 'stale7'): build snapshots then run core."""
    print(f"\n=== Version {label}: max_staleness_days = {max_staleness_days} ===")
    snapshots = _build_snapshots(forecaster_day, max_staleness_days)
    return _run_aggregate_from_snapshots(snapshots, label=label, agg_out_base=agg_out_base)["paper_table"]





# Trimming pipeline that does:
#   1) Choose freeze_day where ~resolved_fraction_at_freeze of questions have resolved.
#   2) Score forecasters on history available by then (questions resolved by freeze_day).
#   3) Trim worst bottom_frac_to_trim (by coverage-adjusted ordered Brier).
#   4) Aggregate excluding trimmed forecasters only AFTER freeze_day.


def run_aggregate_with_trimming(
    forecaster_day: pd.DataFrame,
    *,
    label: str = "stale7_trimmed",
    max_staleness_days: int = 7,
    resolved_fraction_at_freeze: float = 0.40,
    bottom_frac_to_trim: float = 0.10,
    min_days_eligibility: int = 10,
    min_participation_rate: float = 0.10,
    agg_out_base: str = "question_day_means",
):
    
    print("\n=== Version where we rank forecasters and exclude worst from a certain time onwards ===")
    
    # A) Freeze day (~40% resolved)
    if "correctness_known_day" not in forecaster_day.columns:
        # derive from possible timestamp columns
        for ck_col in ("answer correctness known at", "answer correctness_known_at"):
            if ck_col in forecaster_day.columns:
                tmp = pd.to_datetime(forecaster_day[ck_col], utc=True, errors="coerce").dt.floor("D")
                forecaster_day = forecaster_day.copy()
                forecaster_day["correctness_known_day"] = tmp
                break
        else:
            raise ValueError("forecaster_day must include 'correctness_known_day' or a correctness timestamp.")

    q_ck = (
        forecaster_day[["discover question id", "correctness_known_day"]]
        .dropna()
        .drop_duplicates()
        .sort_values("correctness_known_day")
    )
    if q_ck.empty:
        raise ValueError("No resolved questions available to determine a freeze day.")

    idx = max(0, math.ceil(resolved_fraction_at_freeze * len(q_ck)) - 1)
    freeze_day = q_ck["correctness_known_day"].iloc[idx]
    frac_resolved = (q_ck["correctness_known_day"] <= freeze_day).mean()
    print(f"[trim] Freeze day = {freeze_day.date()} (≈{frac_resolved:.0%} resolved)")

    # B) Score individuals on training slice (only questions resolved by freeze day)
    train = forecaster_day[forecaster_day["correctness_known_day"] <= freeze_day].copy()
    if train.empty:
        raise ValueError("No questions resolved by freeze day; cannot rank forecasters.")
    _, per_q_guid, _ = score_individuals(train, max_staleness_days=max_staleness_days)

    # --- Participation rate components (question calendar) ---
    # Ensure datetimes (guard against object dtypes)
    train["day"] = pd.to_datetime(train["day"], utc=True, errors="coerce")
    train["correctness_known_day"] = pd.to_datetime(train["correctness_known_day"], utc=True, errors="coerce")

    # Per-question calendar: start day and correctness-known day
    q_start = (
        train.groupby("discover question id", as_index=False, observed=True)
             .agg(question_start_day=("day", "min"))
    )
    q_total = (
        train.groupby("discover question id", as_index=False, observed=True)
             .agg(correctness_known_day=("correctness_known_day", "max"))
    )
    q_cal = q_start.merge(q_total, on="discover question id", how="inner")
    # Total days in the training window, inclusive
    q_cal["total_days"] = (q_cal["correctness_known_day"] - q_cal["question_start_day"]).dt.days + 1

    # Robustly attach total_days to per_q_guid
    per_q_guid = per_q_guid.copy()
    # Align dtypes on the key (prevents silent non-matches)
    if "discover question id" in per_q_guid.columns and "discover question id" in q_cal.columns:
        per_q_guid["discover question id"] = per_q_guid["discover question id"].astype(q_cal["discover question id"].dtype)

    td_map = q_cal.set_index("discover question id")["total_days"]
    per_q_guid["total_days"] = per_q_guid["discover question id"].map(td_map)

    # Fallback for any NaNs: compute span from observed training days
    if per_q_guid["total_days"].isna().any():
        span = (
            train.groupby("discover question id", as_index=False, observed=True)
                 .agg(span_days=("day", lambda s: (s.max() - s.min()).days + 1))
        )
        span_map = span.set_index("discover question id")["span_days"]
        per_q_guid["total_days"] = per_q_guid["total_days"].fillna(
            per_q_guid["discover question id"].map(span_map)
        )

    # Final guardrails
    per_q_guid["total_days"] = per_q_guid["total_days"].fillna(1).clip(lower=1).astype("float64")
    per_q_guid["participation_rate"] = (per_q_guid["n_days"].astype("float64")
                                        / per_q_guid["total_days"]).clip(0, 1)
    per_q_guid["coverage_adjusted_brier"] = (
        per_q_guid["avg_brier"] / per_q_guid["participation_rate"].replace(0, pd.NA)
    )

    # Eligibility + select bottom 10%
    eligible = per_q_guid[
        (per_q_guid["n_days"] >= min_days_eligibility)
        & (per_q_guid["participation_rate"] >= min_participation_rate)
        & per_q_guid["coverage_adjusted_brier"].notna()
    ].copy()

    if eligible.empty:
        print("[trim] No eligible forecasters to trim. Skipping trimming.")
        excluded_guids = set()
    else:
        eligible_mean = (
            eligible.groupby("membership guid", as_index=False, observed=True)
                    .agg(score=("coverage_adjusted_brier", "mean"))
        )
        n_trim = max(1, math.floor(bottom_frac_to_trim * len(eligible_mean)))
        worst = eligible_mean.sort_values("score", ascending=False).head(n_trim)
        excluded_guids = set(worst["membership guid"])
        print(f"[trim] Excluding {len(excluded_guids)} forecasters (bottom {bottom_frac_to_trim:.0%} performance) from continuation.")
       

    # C) Build snapshots for the whole period; drop excluded AFTER freeze day
    snapshots_full = _build_snapshots(forecaster_day, max_staleness_days)
    if excluded_guids:
        mask_drop = (snapshots_full["day"] > freeze_day) & (snapshots_full["membership guid"].isin(excluded_guids))
        n_before = len(snapshots_full)
        snapshots_full = snapshots_full.loc[~mask_drop].copy()
        print(f"[trim] Dropped {n_before - len(snapshots_full):,} snapshot rows post-freeze for excluded forecasters.")

    # D) Run the shared core on the trimmed snapshots
    core = _run_aggregate_from_snapshots(snapshots_full, label=label, agg_out_base=agg_out_base)
    core.update({"freeze_day": freeze_day, "excluded_guids": excluded_guids})
    return core
