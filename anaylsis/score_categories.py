"""
Task 32: Determine score categories.
Task 33: Implement the score-to-category function.

Thresholds are picked using the ACTUAL score distribution from
task 31 (not arbitrary numbers like 0.7), specifically where
successful vs. failed circuits separate.

Save this as: analysis/score_categories.py
"""

import pandas as pd


def determine_thresholds(scores_df):
    """
    Pick two thresholds using quantiles of the actual score
    distribution, informed by where successful/failed circuits
    tend to land (from task 31's group averages).

    We use the 33rd and 67th percentiles of the overall score
    distribution as a data-driven starting point: this splits
    circuits into three roughly-even groups rather than assuming
    a fixed cutoff. Adjust here if your data suggests otherwise.
    """
    low_threshold = scores_df["trainability_score"].quantile(0.33)
    high_threshold = scores_df["trainability_score"].quantile(0.67)
    return low_threshold, high_threshold


def score_to_category(score, low_threshold, high_threshold):
    """
    Task 33: Convert a Trainability Score into a category label.
    """
    if score >= high_threshold:
        return "Likely Trainable"
    elif score >= low_threshold:
        return "Possibly Difficult"
    else:
        return "Likely Difficult to Train"


if __name__ == "__main__":
    scores = pd.read_csv("data/dev_trainability_scores.csv")

    low_threshold, high_threshold = determine_thresholds(scores)

    print(f"Low threshold (33rd percentile): {low_threshold:.3f}")
    print(f"High threshold (67th percentile): {high_threshold:.3f}")
    print()
    print("Category boundaries:")
    print(f"  score < {low_threshold:.3f}              -> Likely Difficult to Train")
    print(f"  {low_threshold:.3f} <= score < {high_threshold:.3f}  -> Possibly Difficult")
    print(f"  score >= {high_threshold:.3f}             -> Likely Trainable")

    scores["category"] = scores["trainability_score"].apply(
        lambda s: score_to_category(s, low_threshold, high_threshold)
    )

    print("\nCategory counts:")
    print(scores["category"].value_counts())

    print("\nActual success rate WITHIN each category "
          "(sanity check -- categories should track real outcomes):")
    print(scores.groupby("category")["successful"].mean())

    scores.to_csv("data/dev_categorized_scores.csv", index=False)
    print("\nSaved: data/dev_categorized_scores.csv")
