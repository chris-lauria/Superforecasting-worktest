#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from carry_forward import carry_forward_snapshots
from ordered_brier_eval import ordered_brier_from_distribution


def score_individuals(forecaster_day, *, max_staleness_days=None):
    """
    Scores individuals with Ordered Brier, then adjusts by Participation Rate (PR):
      PR = (# active person-days) / (# possible question-days).
    coverage_adjusted_brier = avg_brier / PR
    """

    # STEP 1 — carry-forward per person/day (pass truth + cutoff through)
    snapshots = carry_forward_snapshots(
        forecaster_day,
        max_staleness_days=max_staleness_days,
        per_answer_cols=("resolved_probability",),
        per_question_cols=("correctness_known_day",),
        downcast=True,
    )

    # STEP 2 — trim to day <= correctness_known_day (if available)
    if "correctness_known_day" in snapshots.columns:
        m = snapshots["correctness_known_day"].notna()
        snapshots = snapshots[(~m) | (snapshots["day"] <= snapshots["correctness_known_day"])]

    # STEP 3 — complete bucket vectors per person-day (missing buckets -> 0)
    answers_catalog = (
        forecaster_day[["discover question id", "answer sort order", "resolved_probability"]]
        .drop_duplicates()
        .rename(columns={"answer sort order": "answer_sort_order"})
    )
    snaps = snapshots.rename(columns={"answer sort order": "answer_sort_order"})
    keys = snaps[["discover question id", "membership guid", "day", "correctness_known_day"]].drop_duplicates()
    skeleton = keys.merge(answers_catalog, on="discover question id", how="left")
    skeleton = skeleton.merge(
        snaps[["discover question id", "membership guid", "day", "answer_sort_order", "prob"]],
        on=["discover question id", "membership guid", "day", "answer_sort_order"],
        how="left",
    )
    skeleton["prob"] = skeleton["prob"].fillna(0.0)

    # STEP 4 — Ordered Brier per person-day; then average per (qid, guid)
    by = ["discover question id", "membership guid", "day", "correctness_known_day"]

    def _obrier(g):
        g = g.sort_values("answer_sort_order")
        return ordered_brier_from_distribution(
            g["prob"].to_numpy(dtype=float),
            g["resolved_probability"].to_numpy(dtype=float),
        )

    daily = (
        skeleton.groupby(by, as_index=False)
        .apply(lambda g: pd.Series({"brier": _obrier(g)}))
        .reset_index(drop=True)
        .sort_values(by)
    )

    per_guid_per_question = (
        daily.groupby(["discover question id", "membership guid"], as_index=False)
        .agg(avg_brier=("brier", "mean"), n_days=("brier", "size"))
        .sort_values(["discover question id", "membership guid"])
    )

    # STEP 5 — Participation Rate and coverage-adjusted score
    # Question start day (calendar) ≈ earliest observed day for that question
    q_start = (
        forecaster_day.groupby("discover question id", as_index=False)
        .agg(question_start_day=("day", "min"))
    )
    # Attach cutoff (correctness_known_day) per question
    q_cut = (
        forecaster_day.dropna(subset=["correctness_known_day"])
        .groupby("discover question id", as_index=False)
        .agg(correctness_known_day=("correctness_known_day", "max"))
    )
    q_cal = q_start.merge(q_cut, on="discover question id", how="left")

    # total possible calendar days per question (inclusive)
    q_cal["total_days"] = (
        (q_cal["correctness_known_day"] - q_cal["question_start_day"]).dt.days + 1
    )

    # merge totals onto per-person results
    per_guid_per_question = per_guid_per_question.merge(
        q_cal[["discover question id", "total_days"]],
        on="discover question id",
        how="left",
    )

    # Participation Rate and adjusted score
    per_guid_per_question["participation_rate"] = (
        per_guid_per_question["n_days"] / per_guid_per_question["total_days"]
    )
    per_guid_per_question["coverage_adjusted_brier"] = (
        per_guid_per_question["avg_brier"] / per_guid_per_question["participation_rate"]
    )

    # Per guid overall — average adjusted score across questions
    per_guid_overall = (
        per_guid_per_question.groupby("membership guid", as_index=False)
        .agg(
            avg_brier=("avg_brier", "mean"),
            avg_adjusted_brier=("coverage_adjusted_brier", "mean"),
            n_questions=("discover question id", "nunique"),
            n_days=("n_days", "sum"),
            mean_participation=("participation_rate", "mean"),
        )
        .sort_values("membership guid")
    )

    return daily, per_guid_per_question, per_guid_overall
