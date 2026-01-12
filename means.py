#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


# small constant to prevent log(0) or div-by-zero when forecasters predict with
#certainty

EPS = 1e-5  


def geometric_mean_prob(probs):
    """
    Geometric mean of probabilities.
    For probs p1, p2, ..., pn this returns (p1 * p2 * ... * pn) ** (1/n),
    computed in log space for numerical stability.

    Rules:
    - If any probability is 0, returns 0.0
    - If any probability is <0, returns NaN (invalid for a probability)
    """
    arr = np.asarray(probs, dtype=float)

    # basic validation
    if arr.size == 0:
        return np.nan
    if np.any(arr < 0):
        return np.nan

    
    # clip to avoid 0 and 1 edge issues
    arr = np.clip(arr, EPS, 1 - EPS)

    # standard geometric mean using log/exp
    # geo_mean = exp(mean(log(p)))
    return float(np.exp(np.log(arr).mean()))


def trimmed_mean_prob(probs, trim_frac=0.1):
    """
    Trimmed mean of probabilities.
    Drops the lowest trim_frac * 100% and the highest trim_frac * 100%
    of the values, then averages what's left.

    Example: trim_frac = 0.1 → drop bottom 10% and top 10%.

    Behavior:
    - If there aren't enough values to trim both ends (like very small n),
      it just won't over-trim; it'll fall back to using whatever remains.
    - Returns NaN if no values.
    """
    arr = np.asarray(probs, dtype=float)

    if arr.size == 0:
        return np.nan

    # sort ascending
    arr = np.sort(arr)

    # how many to cut off each tail
    k = int(np.floor(arr.size * trim_frac))
    
    #print(k)

    # if trimming both sides would delete everything, don't trim
    if k * 2 >= arr.size:
        trimmed = arr
    else:
        trimmed = arr[k: arr.size - k]

    return float(trimmed.mean())


def geometric_mean_odds(probs):
    """
    Geometric mean of the *odds*, where odds = p / (1 - p).

    Steps:
    1. Convert each probability p to odds o = p / (1 - p).
    2. Take the geometric mean of those odds.

    Rules:
    - If any p == 1.0, odds is infinite → return inf.
    - If any p == 0.0, odds is 0 → geometric mean of odds is 0.
    - If any p is outside [0,1], return NaN.

    Returned value is the geometric mean of odds (not converted back to a probability).
    """
    arr = np.asarray(probs, dtype=float)

    if arr.size == 0:
        return np.nan

    # probabilities must be between 0 and 1
    if np.any(arr < 0) or np.any(arr > 1):
        return np.nan


    # clip to avoid 0 and 1 edge issues
    arr = np.clip(arr, EPS, 1 - EPS)


    # convert to odds
    odds = arr / (1.0 - arr)
    
    #print(odds)
    
    
    # geometric mean of the odds (log-space for stability)
    o_geo = float(np.exp(np.log(odds).mean()))

    # convert geometric-mean odds back to probability
    p_geo = o_geo / (1.0 + o_geo)
    

    # geometric mean of odds, same log/exp trick
    return float(p_geo)



# test =[2,2,4,8,2,0,0,8,0,8]



# print(trimmed_mean_prob(test))


# test1 =[0.1, 0.2]


# print(geometric_mean_odds(test1))













