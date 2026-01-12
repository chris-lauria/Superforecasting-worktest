#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import pandas as pd


#This function is needed to keep things more efficient because otherwise
# the kernel gets killed on my laptop, it carries forward forecasts for scoring as
# is canonical to do in these kinds of settings

def carry_forward_snapshots(
    forecaster_day,
    *,
    max_staleness_days=None,
    per_answer_cols=("resolved_probability",),
    per_question_cols=(),
    downcast=True,
):
    """
    Build per-day carried-forward snapshots from a compact forecaster_day table,
    emitting chunks per question to keep peak memory low.
    """
    pa_cols = [c for c in (per_answer_cols or ()) if c in forecaster_day.columns]
    pq_cols = [c for c in (per_question_cols or ()) if c in forecaster_day.columns]

    # Per-(qid, answer) lookups (tiny)
    per_answer_lookup = {}
    if pa_cols:
        uniq_pa = forecaster_day.drop_duplicates(
            ["discover question id", "answer sort order"] + list(pa_cols)
        )
        for _, row in uniq_pa.iterrows():
            key = (row["discover question id"], row["answer sort order"])
            per_answer_lookup[key] = {c: row[c] for c in pa_cols}

    # Per-question lookups (tiny)
    per_question_lookup = {}
    if pq_cols:
        uniq_pq = forecaster_day.drop_duplicates(["discover question id"] + list(pq_cols))
        for _, row in uniq_pq.iterrows():
            per_question_lookup[row["discover question id"]] = {c: row[c] for c in pq_cols}

    out_chunks = []

    # Process per question to cap memory
    for qid, sub_q in forecaster_day.groupby("discover question id", sort=False):
        rows_q = []

        days_sorted = sorted(pd.unique(sub_q["day"]))

        latest_probs = {}  # (guid, aso) -> prob
        latest_day = {}    # (guid, aso) -> last update day

        for day in days_sorted:
            today_rows = sub_q[sub_q["day"] == day]

            # Update state with today's submissions (last-write-wins within the day)
            
            for (guid, aso), block in today_rows.groupby(
                ["membership guid", "answer sort order"], sort=False, observed=True
            ):
                latest_probs[(guid, aso)] = float(block["prob"].iloc[-1])
                latest_day[(guid, aso)] = day

            # Drop stale if requested
            if max_staleness_days is not None:
                cutoff = pd.Timedelta(days=max_staleness_days)
                to_drop = [
                    k for k, dlast in latest_day.items()
                    if (pd.Timestamp(day) - pd.Timestamp(dlast)) > cutoff
                ]
                for k in to_drop:
                    latest_day.pop(k, None)
                    latest_probs.pop(k, None)

            # Emit active entries for this day
            for (guid, aso), p in latest_probs.items():
                rec = {
                    "discover question id": qid,
                    "membership guid": guid,
                    "day": day,
                    "answer sort order": aso,
                    "prob": p,
                }
                if pa_cols:
                    rec.update(per_answer_lookup.get((qid, aso), {}))
                if pq_cols:
                    rec.update(per_question_lookup.get(qid, {}))
                rows_q.append(rec)

        if rows_q:
            chunk = pd.DataFrame.from_records(rows_q)

            if downcast:
                # numeric downcast
                chunk["prob"] = chunk["prob"].astype("float32")
                for c in pa_cols + pq_cols:
                    if c in chunk and pd.api.types.is_float_dtype(chunk[c]):
                        chunk[c] = chunk[c].astype("float32")
                # ids as categoricals / small ints
                for c in ["discover question id", "membership guid"]:
                    if chunk[c].dtype == object:
                        chunk[c] = chunk[c].astype("category")
                if pd.api.types.is_integer_dtype(chunk["answer sort order"]):
                    chunk["answer sort order"] = chunk["answer sort order"].astype("int16")

            out_chunks.append(chunk)

    if not out_chunks:
        cols = [
            "discover question id", "membership guid", "day", "answer sort order", "prob"
        ] + pa_cols + pq_cols
        return pd.DataFrame(columns=cols)

    # Concatenate per-question chunks
    snapshots = pd.concat(out_chunks, ignore_index=True)
    return snapshots
