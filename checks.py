#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#After an inspection with bash commands I believe the data is quite clean so I will be letting
#pandas infer the datatype on its own to keep the code light instead of parsing everything
#as strings


import pandas as pd

FILE = "rct-a-prediction-sets.csv"
SUM_TOL = 1e-3  # tolerance for sum-to-one

def main(path=FILE):
    # Let pandas infer dtypes; keep defaults for NA handling
    df = pd.read_csv(path)

    #  strip removes any leading or trailing whitespace in the column names
    df.rename(columns=lambda c: str(c).strip(), inplace=True)

    # Required columns
    req = [
        "prediction set id",
        "discover question id",         
        "answer id",
        "forecasted probability",
        "filled at",
        "answer resolved probability",   
        "membership guid",               
        "answer sort order",             
    ]

    missing = [c for c in req if c not in df.columns]
    if missing:
        print("FATAL: missing required columns:", ", ".join(missing))
        return

    # 1) Nulls in required fields (clean data => mostly NaN-based)
    print("\n[Nulls in required fields]")
    for c in req:
        n = int(df[c].isna().sum())
        print(f"- {c}: {n}")
        
        
    qids = df.loc[df["answer resolved probability"].isna(), "discover question id"].dropna().unique().tolist()
    print(qids)
  
    #191 This question was voided due to an error in the answer thresholds. 
        


    
    # 2) Probability range [0,1] 
    print("\n[Probability outside 0..1]")
    fp = df["forecasted probability"]

    if not pd.api.types.is_numeric_dtype(fp):
        print(f"FATAL: 'forecasted probability' is not numeric (dtype={fp.dtype}).")
        return

    bad_prob = df[fp.notna() & ~fp.between(0, 1)]
    print(f"- rows: {len(bad_prob)}")


    # 3) Duplicate forecast events (same set/answer/time)
    print("\n[Duplicate (prediction set id, answer id, filled at)]")
    key = ["prediction set id", "answer id", "filled at"]
    dup_mask = df.duplicated(subset=key, keep=False)
    print(f"- rows: {int(dup_mask.sum())}")
    
    
   # 4) checking if answer sort order per prediction set is always {0..n-1} 
    
    grp = df.groupby("prediction set id")["answer sort order"]
    
    def ok_set(s):
        
        if s.isna().any():
            return False
        uniq = set(s)             
        n = len(uniq)
        if not (1 <= n <= 5):
            return False
        return uniq == set(range(n))  # exactly {0}, {0,1}, ..., {0,1,2,3,4}
    
    res = grp.apply(ok_set).reset_index(name="ok")
    
    print("\n[answer sort order per prediction question id]")
    print(f"OK: {int(res['ok'].sum())}/{len(res)}  anomalies: {int((~res['ok']).sum())}")
    
    if (~res["ok"]).any():
        # show raw distinct values (as a set) for quick inspection — no sorting
        bad = res[~res["ok"]].merge(
            grp.apply(lambda s: set(s)).reset_index(name="distinct_raw"),
            on="prediction question id"
        )
        print(bad.head(10).to_string(index=False))


    # 4) Sum-to-one across answers per prediction set (exclude single-answer sets)
    print("\n[Sum-to-one by prediction set id (excluding single-answer sets)]")
    
    tmp = df.copy()
    tmp["__fp__"] = fp
    
    # Keep only prediction sets that have >1 DISTINCT answers
    if "answer id" in tmp.columns:
        multi = tmp.groupby("prediction set id")["answer id"].transform("nunique") > 1
    else:
        # Fallback: if 'answer id' missing, use row count
        multi = tmp.groupby("prediction set id")["__fp__"].transform("size") > 1
    
    tmp = tmp[multi]
    
    sums = (
        tmp.groupby("prediction set id", dropna=False)["__fp__"]
           .sum(min_count=1)
           .reset_index(name="sum_prob")
    )
    
    off = sums[sums["sum_prob"].notna() & ((sums["sum_prob"] - 1.0).abs() > SUM_TOL)]
    print(f"- events off 1.0: {len(off)}")
    if not off.empty:
        print(off.head(5).to_string(index=False))


    
    # 5) Optional: created at <= updated at (parse only if columns exist)
    if {"created at", "updated at"}.issubset(df.columns):
        created = pd.to_datetime(df["created at"], errors="coerce", utc=True)
        updated = pd.to_datetime(df["updated at"], errors="coerce", utc=True)
        bad_time = df[created.notna() & updated.notna() & (created > updated)]
        print("\n[created at <= updated at]")
        print(f"- rows: {len(bad_time)}")

    print("\nDone.")

if __name__ == "__main__":
    main()

