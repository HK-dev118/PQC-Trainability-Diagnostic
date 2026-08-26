"""
Task 37: Identify the most useful indicators.

Combines task 30's correlation strength with task 34's individual
predictive accuracy into one ranked table, so it's clear which
indicators actually matter and which contribute little.

Save this as: analysis/rank_indicators.py
"""

import pandas as pd


ALL_INDICATORS = [
    "mean_gradient_magnitude",
    "gradient_variance",
    "near_zero_percentage",
    "n_qubits",
    "depth",
    "n_trainable_parameters",
    "n_gates",
]

# From task 34's corrected individual-indicator run.
INDIVIDUAL_ACCURACY = {
    "n_qubits": 0.854,
    "mean_gradient_magnitude": 0.787,
    "gradient_variance": 0.787,
    "near_zero_percentage": 0.679,
    "n_trainable_parameters": 0.673,
    "n_gates": 0.673,
    "depth": 0.513,
}


def classify_usefulness(accuracy):
    if accuracy >= 0.75:
        return "Strong"
    elif accuracy >= 0.60:
        return "Moderate"
    else:
        return "Weak (near coin-flip)"


if __name__ == "__main__":
    correlations = pd.read_csv("data/task30_correlations.csv")

    ranking = []
    for indicator in ALL_INDICATORS:
        corr_row = correlations[correlations["indicator"] == indicator].iloc[0]
        ranking.append({
            "indicator": indicator,
            "individual_accuracy": INDIVIDUAL_ACCURACY[indicator],
            "correlation_with_success": corr_row["correlation_with_success"],
            "usefulness": classify_usefulness(
                INDIVIDUAL_ACCURACY[indicator]
            ),
        })

    ranking_df = pd.DataFrame(ranking).sort_values(
        "individual_accuracy", ascending=False
    ).reset_index(drop=True)
    ranking_df.index = ranking_df.index + 1  # rank starting at 1

    print("=== Task 37: Indicator ranking ===\n")
    print(ranking_df.to_string())

    print("\n=== Summary ===")
    strong = ranking_df[ranking_df["usefulness"] == "Strong"]["indicator"].tolist()
    moderate = ranking_df[ranking_df["usefulness"] == "Moderate"]["indicator"].tolist()
    weak = ranking_df[ranking_df["usefulness"] == "Weak (near coin-flip)"]["indicator"].tolist()

    print(f"Strong indicators: {', '.join(strong)}")
    print(f"Moderate indicators: {', '.join(moderate)}")
    print(f"Weak indicators: {', '.join(weak)}")

    ranking_df.to_csv("data/task37_indicator_ranking.csv")
    print("\nSaved: data/task37_indicator_ranking.csv")
