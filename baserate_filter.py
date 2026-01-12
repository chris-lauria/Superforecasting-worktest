#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This code filters the rationale column with a regular expression for the words
# below

import re

# Matches: "base rate(s)", "base-rate(s)", "reference class(es)", "reference-class(es)"
KEYWORD_RE = re.compile(r"(?i)\bbase[-\s]?rates?\b|\breference[-\s]?class(?:es)?\b")


def keyword_guids(
    df,
    *,
    rationale_col="rationale",
    guid_col="membership guid",
    pattern=KEYWORD_RE,
    min_hits=1,
):
    """Return set of guids that used the keywords at least `min_hits` times."""
    if rationale_col not in df.columns:
        raise ValueError(f"Missing column: {rationale_col!r}")
    if guid_col not in df.columns:
        raise ValueError(f"Missing column: {guid_col!r}")

    s = df[rationale_col].astype(str).fillna("")
    hits = s.str.contains(pattern, regex=True, na=False)
    counts = df.assign(_hit=hits).groupby(guid_col, observed=True)["_hit"].sum()
    return set(counts[counts >= min_hits].index)


def filter_by_rationale(
    df,
    *,
    rationale_source,
    rationale_col="rationale",
    guid_col="membership guid",
    pattern=KEYWORD_RE,
    min_hits=1,
):
    """
    Filter `df` to rows whose guid is in the set of forecasters that used the
    keywords at least `min_hits` times in `rationale_source`.
    Prints the percentage of forecasters removed.
    """
    keep = keyword_guids(
        rationale_source,
        rationale_col=rationale_col,
        guid_col=guid_col,
        pattern=pattern,
        min_hits=min_hits,
    )

    # Diagnostics: percent of forecasters removed
    total_guids = rationale_source[guid_col].nunique()
    kept_guids = len(keep)
    removed_guids = total_guids - kept_guids
    if total_guids > 0:
        pct_removed = removed_guids / total_guids * 100.0
        print(f"[rationale_filter] Removed {removed_guids} of {total_guids} forecasters ({pct_removed:.2f}%).")
    else:
        print("[rationale_filter] No forecasters found to filter.")

    return df[df[guid_col].isin(keep)].copy()
