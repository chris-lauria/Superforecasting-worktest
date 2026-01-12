#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

def build_paper_table(day_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-day scores into a compact table for the paper.
    Rows = aggregation methods.
    Cols = mean ordered Brier score, std dev, N.
    Lower score = better.
    """

    methods = {
        "obs_mean":        "Mean",
        "obs_median":      "Median",
        "obs_trimmed":     "Trimmed mean (10%)",
        "obs_geom_prob":   "Geometric mean (prob)",
        "obs_geom_odds":   "Geometric mean (odds)",
    }

    rows = []
    for col, pretty_name in methods.items():
        vals = day_scores[col].to_numpy(dtype=float)

        rows.append({
            "Method": pretty_name,
            "Ordered Brier (avg)": np.mean(vals),
            "Std dev across days": np.std(vals, ddof=1),
            "N question-days": vals.size,
        })

    out = pd.DataFrame(rows)

    # sort by best (lowest avg score)
    out = out.sort_values("Ordered Brier (avg)").reset_index(drop=True)

    return out
