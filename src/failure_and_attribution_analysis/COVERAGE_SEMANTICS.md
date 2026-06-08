# Coverage Semantics

This project uses `coverage` as an exploration-progress metric, not as a count of
all possible parameter combinations.

## What coverage means

- Samples are clustered into local regions in the continuous scenario space.
- Each region is checked for:
  - `RAU`: average predictive uncertainty in the region.
  - `SC`: sampling completeness / density in the region.
- A region is treated as covered when it no longer needs exploration.
- Global coverage is the sample-weighted ratio of covered samples to total samples.

## What coverage does not mean

- It does not directly measure whether all attack types are evenly represented.
- It does not mean every discrete attack setting has been fully explored.
- It does not represent geometric coverage of the full Cartesian product of parameters.

## Interpretation by mode

- Default flow:
  Coverage means conditional coverage under the current exploration constraints,
  mainly over the continuous scenario parameters.
- Single-attack flow:
  Coverage means conditional coverage of the continuous scenario space with the
  selected attack type fixed.

## Current early-stop rule

- The runtime currently stops early when `coverage_upper_bound >= coverage_target`
  and the minimum sample count has been reached.
- This is a practical stopping rule for exploration progress.
- It is less conservative than requiring `coverage_lower_bound >= coverage_target`.

