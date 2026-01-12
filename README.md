# Superforecasting-worktest

.------. .------. .------. .------. .------. .------.
|  R   | |  E   | |  A   | |  D   | |  M   | |  E   |
'------' '------' '------' '------' '------' '------'


This repo contains the code I utilized to pass the a worktest for a research position at a superforecasting company

To run the script, execute main.py. It will display in the console the results.

The other Python files are modules used by main.py, except for checks.py, which was only used to perform some data verification.

data_cleaning.py: drops unresolved questions and post-resolution forecasts.

carry_forward.py: carries forward forecasts as is canonically done in competitions, this is computationally expensive so the code needed to be streaming.

aggregation.py: computes daily per-answer aggregates.

ordered_brier_eval.py: computes ordered Brier scores.

individual_scoring.py: scores individuals trying to take of the fact that they could have entered questions at different times (questions are easier towards when the answer will 
be known).

build_table.py: builds a compact table for reporting.

pipeline1.py:  Runs different variants of aggregating forecasts to try and see if one is better.  

baserate_filter.py:  Filters individual forecasters that ever used the words "base rate" or "reference class" once in their rational
