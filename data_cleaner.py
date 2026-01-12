#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def drop_unresolved_questions(df):
    """
    Clean the raw prediction rows.

    Steps
    -----
    1. Identify any discover question id where at least one row has
       'answer resolved probability' = NaN. We drop *all* rows for those
       questions, because we can't score them.
    2. From the remaining rows, drop any forecasts that were made AFTER
       the answer was already known. The column 'made after correctness known'
       marks those. Those predictions should not count.

    Returns
    -------
    cleaned : DataFrame
        Filtered copy of df after both steps.
    removed_qids : list
        The list of discover question ids we dropped entirely in step 1.
    n_after_known_removed : int
        How many rows we dropped in step 2 because they were made after
        correctness was known.
    """

    # ---------- Step 1: drop entire unresolved questions ----------
    bad_qids = (
        df.loc[df["answer resolved probability"].isna(), "discover question id"]
        .dropna()
        .unique()
        .tolist()
    )

    keep_q_mask = ~df["discover question id"].isin(bad_qids)
    df_step1 = df[keep_q_mask].copy()

    # ---------- Step 2: drop post-resolution (after-known) forecasts ----------
    # "made after correctness known" may be bools or strings.
    raw_col = df_step1["made after correctness known"]

    normalized_after_known = (
        raw_col
        .astype(str)  # True -> "True", NaN -> "nan", etc.
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )

    # Now:
    # "true" -> True
    # "false" -> False
    # anything else (including "nan") -> NaN
    df_step1 = df_step1.assign(_made_after_known=normalized_after_known)

    # rows to *drop* are those where _made_after_known is True
    drop_mask = df_step1["_made_after_known"] == True
    n_after_known_removed = int(drop_mask.sum())

    # keep rows that are NOT marked True
    cleaned = (
        df_step1[~drop_mask]
        .drop(columns=["_made_after_known"])
        .copy()
    )

    return cleaned, bad_qids, n_after_known_removed
