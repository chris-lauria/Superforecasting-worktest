#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pandas as pd
from data_cleaner import drop_unresolved_questions
from baserate_filter import filter_by_rationale

from pipeline import run_aggregate_variant, run_aggregate_with_trimming


INPUT_FILE = "rct-a-prediction-sets.csv"




def main():
    # -----------------------------------------------------------
    # 1. Load csv + drop unresolved questions and rows where the forecast was made late
    # -----------------------------------------------------------
    df = pd.read_csv(INPUT_FILE)
    cleaned_df, removed_ids, n_after_known_removed = drop_unresolved_questions(df)

    # Cleanup summary prints
    n_rows_dropped_unresolved_q = int(df["discover question id"].isin(removed_ids).sum())
    print(
        f"Removed {len(removed_ids)} discover question ids with unresolved answers "
        f"({n_rows_dropped_unresolved_q} rows):"
    )
    for q in removed_ids:
        print(f"- {q}")
    print(f"Also dropped {n_after_known_removed} forecasts made after correctness was known.")
    print(f"Final row count after cleaning: {len(cleaned_df)} (from {len(df)} original)")

    # -------------------------------------------------
    # 2. Parse timestamps and define 'day' as the floor of created at
    # -------------------------------------------------
    cleaned_df["created at"] = pd.to_datetime(cleaned_df["created at"], utc=True, errors="coerce")
    cleaned_df["day"] = cleaned_df["created at"].dt.floor("D")

    # -------------------------------------------------
    # 3. Get each forecaster's final prediction set per day as instructed
    # -------------------------------------------------
    sets = (
        cleaned_df[
            ["prediction set id", "discover question id", "membership guid", "day", "created at"]
        ]
        .drop_duplicates()
        .sort_values(
            ["discover question id", "membership guid", "day", "created at", "prediction set id"]
        )
    )
    latest_sets = sets.groupby(["discover question id", "membership guid", "day"], as_index=False).tail(1)
    latest_set_ids = set(latest_sets["prediction set id"])
    latest_rows = cleaned_df[cleaned_df["prediction set id"].isin(latest_set_ids)].copy()

    # -------------------------------------------------
    # 4. Creating the dataframe we will work with (forecaster_day)
    #    + attach correctness_known_day defensively 
    # -------------------------------------------------
   
    cols = [
        "discover question id",
        "membership guid",
        "day",
        "answer sort order",
        "forecasted probability",
        "answer resolved probability",
        "answer correctness_known_at",
    ]
    forecaster_day = latest_rows[cols].rename(
        columns={
            "forecasted probability": "prob",
            "answer resolved probability": "resolved_probability",
        }
    ).copy()
    
    #There are 5 questions with 2 correctness known at dates, we make the call
    #of keeping this data and keeping the earliest of the two correctness known
    #at dates. This is what the code below does .
    
    # Parse correctness-known timestamps and take the EARLIEST per question
    ck_day = pd.to_datetime(
        forecaster_day["answer correctness_known_at"], utc=True, errors="coerce"
    ).dt.floor("D")
    
    q_earliest_ck = (
        forecaster_day.assign(_ck_day=ck_day)
        .dropna(subset=["_ck_day"])
        .groupby("discover question id", observed=True)["_ck_day"]
        .min()
    )
    
    # Attach earliest correctness_known_day to every row of that question
    forecaster_day["correctness_known_day"] = forecaster_day["discover question id"].map(q_earliest_ck)
    
    # Optional: drop the raw timestamp column now that the per-question day is attached
    forecaster_day.drop(columns=["answer correctness_known_at"], inplace=True)



    # -------------------------------------------------
    # 5. Here we Run first a simple aggregator and then one that removes
    # stale forecasts (older than a week)
    # -------------------------------------------------
    
    #options to print tables generated entirely to the console
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)   # prevents line wrapping
    pd.set_option('display.max_colwidth', None)
    
    
    paper_table_nostale = run_aggregate_variant(
        forecaster_day, label="nostale", max_staleness_days=None
    )
    paper_table_stale7 = run_aggregate_variant(
        forecaster_day, label="stale7", max_staleness_days=7
    )

    # -------------------------------------------------
    # 6. Here we run another aggregator that after 40% of the questions
    # have been asked freezes the competition,
    # scores individual forecasters and removes
    # the bottom 10% from further use
    # -------------------------------------------------
    trim_results = run_aggregate_with_trimming(
        forecaster_day=forecaster_day,
        label="stale7_trimmed",
        max_staleness_days=7,
        resolved_fraction_at_freeze=0.40,
        bottom_frac_to_trim=0.10,
        min_days_eligibility=10,
        min_participation_rate=0.10,
    )
    
    
    
    # -------------------------------------------------
    # 7. Lastly try filtering for all people who said "base rate" or
    #  "reference class" once in their rational and aggregate
    #   only their answers
    # -------------------------------------------------
  
    print("\n=== Trying filtering by rational ===")
  
    forecaster_day = filter_by_rationale(
        forecaster_day,
        rationale_source=cleaned_df, 
        rationale_col="rationale",
        guid_col="membership guid",
    )
    
    paper_table_nostale = run_aggregate_variant(forecaster_day, label="nostale", max_staleness_days=None)
    
        
    

    # -------------------------------------------------
    # 8. Final side-by-side comparison 
    # -------------------------------------------------
    pt_nostale = paper_table_nostale[["Method", "Ordered Brier (avg)"]].rename(
        columns={"Ordered Brier (avg)": "No staleness avg score"}
    )
    pt_stale7 = paper_table_stale7[["Method", "Ordered Brier (avg)"]].rename(
        columns={"Ordered Brier (avg)": "7-day cutoff avg score"}
    )
    pt_trim = trim_results["paper_table"][["Method", "Ordered Brier (avg)"]].rename(
        columns={"Ordered Brier (avg)": "7-day + trim avg score"}
    )
    
    comparison = (
        pt_nostale
        .merge(pt_stale7, on="Method", how="outer")
        .merge(pt_trim, on="Method", how="outer")
    )
    
    print("\n=== FINAL COMPARISON (nostale vs stale7 vs stale7_trimmed) ===")
    print(comparison)
    comparison.to_csv("comparison_nostale_vs_stale7_vs_trimmed.csv", index=False)
    
    
    

if __name__ == "__main__":
    main()