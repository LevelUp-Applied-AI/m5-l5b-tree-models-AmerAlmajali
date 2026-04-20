"""
Module 5 Week B — Applied Lab: Trees & Ensembles

Build and evaluate decision tree and random forest models on the Petra
Telecom churn dataset. Handle class imbalance honestly (class_weight as an
operating-point tool at a fixed threshold), evaluate with PR-AUC and
calibration, and demonstrate what tree models capture that linear models
cannot.

Complete the 12 functions below. See the lab guide for task-by-task detail.
Run with:  python lab_trees.py
Tests:     pytest tests/ -v
"""

import os

# Use a non-interactive matplotlib backend so plots save cleanly in CI
# and on headless environments.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "contract_months",
]


# ---------------------------------------------------------------------------
# Task 1 — Load and split
# ---------------------------------------------------------------------------


def load_and_split(filepath="data/telecom_churn.csv", random_state=42):
    """Load the Petra Telecom dataset and split 80/20 with stratification.

    Args:
        filepath: Path to telecom_churn.csv.
        random_state: Random seed for reproducible split.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) where X contains only
        NUMERIC_FEATURES and y is the `churned` column.
    """
    df = pd.read_csv(filepath)
    X = df[NUMERIC_FEATURES]
    y = df["churned"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)


# ---------------------------------------------------------------------------
# Task 2 — Decision tree + calibration comparison
# ---------------------------------------------------------------------------


def build_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """Train a DecisionTreeClassifier.

    Args:
        max_depth: Maximum tree depth (None means unconstrained).
        random_state: Random seed.

    Returns:
        Fitted DecisionTreeClassifier.
    """
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error using equal-count (quantile) binning.

    Sort samples by predicted probability, split into `n_bins` equal-size
    chunks, and sum the bin-weighted absolute difference between each bin's
    mean predicted probability and its fraction of true positives.

    Args:
        y_true: 1D array-like of true binary labels (0 or 1).
        y_prob: 1D array-like of predicted probabilities for class 1.
        n_bins: Number of equal-count bins.

    Returns:
        ECE as a float in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    # Sort by ascending predicted probability
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    # Split indices into n_bins equal-count chunks
    bins = np.array_split(np.arange(n), n_bins)

    ece = 0.0
    for bin_indices in bins:
        if len(bin_indices) == 0:
            continue
        bin_size = len(bin_indices)
        mean_prob = y_prob_sorted[bin_indices].mean()
        frac_positive = y_true_sorted[bin_indices].mean()
        ece += (bin_size / n) * abs(mean_prob - frac_positive)

    return ece


def compare_dt_calibration(X_train, X_test, y_train, y_test):
    """Compare calibration of an unbounded DT vs a depth-5 DT.

    Returns:
        Dict with keys 'ece_unbounded' and 'ece_depth_5' (floats in [0, 1]).
    """
    # Unbounded tree — pure leaves → extreme probabilities → poor calibration
    dt_unbounded = DecisionTreeClassifier(max_depth=None, random_state=42)
    dt_unbounded.fit(X_train, y_train)
    prob_unbounded = dt_unbounded.predict_proba(X_test)[:, 1]
    ece_unbounded = compute_ece(y_test, prob_unbounded)

    # Depth-5 tree — leaves contain multiple samples → smoother probabilities
    dt_depth5 = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_depth5.fit(X_train, y_train)
    prob_depth5 = dt_depth5.predict_proba(X_test)[:, 1]
    ece_depth5 = compute_ece(y_test, prob_depth5)

    return {"ece_unbounded": ece_unbounded, "ece_depth_5": ece_depth5}


# ---------------------------------------------------------------------------
# Task 3 — Random forest + feature importances
# ---------------------------------------------------------------------------


def build_random_forest(
    X_train, y_train, n_estimators=100, max_depth=10, class_weight=None, random_state=42
):
    """Train a RandomForestClassifier.

    Args:
        class_weight: None for default, 'balanced' to reweight the loss.
        random_state: Random seed.

    Returns:
        Fitted RandomForestClassifier.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def get_feature_importances(model, feature_names):
    """Return a dict of feature_name -> importance, sorted descending."""
    pairs = zip(feature_names, model.feature_importances_)
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return dict(sorted_pairs)


# ---------------------------------------------------------------------------
# Task 4 — class_weight at the default 0.5 threshold
# ---------------------------------------------------------------------------


def evaluate_recall_at_threshold(model, X_test, y_test, threshold=0.5):
    """Recall for class 1 at a specified decision threshold.

    Returns:
        Recall as a float in [0, 1].
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return recall_score(y_test, y_pred, zero_division=0)


def compute_pr_auc(model, X_test, y_test):
    """PR-AUC (average precision) for the positive class.

    Returns:
        Float in [0, 1].
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, y_prob)


# ---------------------------------------------------------------------------
# Task 5 — PR curves and calibration curves
# ---------------------------------------------------------------------------


def plot_pr_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot PR curves for both RF models on the same axes and save as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))

    PrecisionRecallDisplay.from_estimator(
        rf_default, X_test, y_test, name="RF default", ax=ax
    )
    PrecisionRecallDisplay.from_estimator(
        rf_balanced, X_test, y_test, name="RF balanced", ax=ax
    )

    ax.set_title("Precision-Recall Curves: Default vs Balanced RF")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot calibration curves for both RF models and save as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))

    CalibrationDisplay.from_estimator(
        rf_default, X_test, y_test, n_bins=10, name="RF default", ax=ax
    )
    CalibrationDisplay.from_estimator(
        rf_balanced, X_test, y_test, n_bins=10, name="RF balanced", ax=ax
    )

    ax.set_title("Calibration Curves: Default vs Balanced RF")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 6 — Tree-vs-linear capability demonstration
# ---------------------------------------------------------------------------


def build_logistic_regression(X_train_scaled, y_train, random_state=42):
    """Train a LogisticRegression baseline on scaled features.

    Returns:
        Fitted LogisticRegression(max_iter=1000).
    """
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train_scaled, y_train)
    return clf


def find_tree_vs_linear_disagreement(
    rf_model, lr_model, X_test_raw, X_test_scaled, y_test, feature_names, min_diff=0.15
):
    """Find ONE test sample where RF and LR predicted probabilities differ most.

    Returns:
        Dict with keys: sample_idx, feature_values, rf_proba, lr_proba,
        prob_diff, true_label. Returns None if no sample exceeds min_diff.
    """
    rf_proba = rf_model.predict_proba(X_test_raw)[:, 1]
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    abs_diff = np.abs(rf_proba - lr_proba)
    max_idx = int(np.argmax(abs_diff))

    if abs_diff[max_idx] < min_diff:
        return None

    # Convert X_test_raw to a numpy array regardless of whether it's a
    # DataFrame or ndarray, so indexing is always consistent.
    X_raw_arr = np.asarray(X_test_raw)
    y_arr = np.asarray(y_test)

    feature_values = {
        name: float(X_raw_arr[max_idx, i]) for i, name in enumerate(feature_names)
    }

    return {
        "sample_idx": max_idx,
        "feature_values": feature_values,
        "rf_proba": float(rf_proba[max_idx]),
        "lr_proba": float(lr_proba[max_idx]),
        "prob_diff": float(abs_diff[max_idx]),
        "true_label": int(y_arr[max_idx]),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main():
    """Orchestrate all 7 lab tasks. Run with: python lab_trees.py"""
    os.makedirs("results", exist_ok=True)

    # Task 1: Load + split
    result = load_and_split()
    if not result:
        print("load_and_split not implemented. Exiting.")
        return
    X_train, X_test, y_train, y_test = result
    print(
        f"Train: {len(X_train)}  Test: {len(X_test)}  Churn rate: {y_train.mean():.2%}"
    )

    # Task 2: Decision tree + calibration comparison
    dt = build_decision_tree(X_train, y_train)
    if dt is not None:
        print(f"\n--- Decision Tree (max_depth=5) ---")
        print(classification_report(y_test, dt.predict(X_test), zero_division=0))
        # Plot tree (first 3 levels)
        plt.figure(figsize=(18, 10))
        plot_tree(
            dt, feature_names=NUMERIC_FEATURES, max_depth=3, filled=True, fontsize=8
        )
        plt.savefig("results/decision_tree.png", dpi=100, bbox_inches="tight")
        plt.close()

    cal = compare_dt_calibration(X_train, X_test, y_train, y_test)
    if cal:
        print(f"DT ECE (max_depth=None): {cal['ece_unbounded']:.3f}")
        print(f"DT ECE (max_depth=5):    {cal['ece_depth_5']:.3f}")

    # Task 3: Random forest + feature importances
    rf = build_random_forest(X_train, y_train)
    if rf is not None:
        print(f"\n--- Random Forest (max_depth=10) ---")
        imp = get_feature_importances(rf, NUMERIC_FEATURES)
        if imp:
            print("Feature importances:")
            for name, value in imp.items():
                print(f"  {name:<22s} {value:.3f}")

    # Task 4: Balanced RF + recall@0.5 comparison + PR-AUC
    rf_bal = build_random_forest(X_train, y_train, class_weight="balanced")
    if rf is not None and rf_bal is not None:
        r_def = evaluate_recall_at_threshold(rf, X_test, y_test, threshold=0.5)
        r_bal = evaluate_recall_at_threshold(rf_bal, X_test, y_test, threshold=0.5)
        print(f"\n--- class_weight effect at default 0.5 threshold ---")
        print(f"  RF default recall@0.5:  {r_def:.3f}")
        print(
            f"  RF balanced recall@0.5: {r_bal:.3f}  (ratio: {r_bal / max(r_def, 1e-9):.2f}x)"
        )

        auc_def = compute_pr_auc(rf, X_test, y_test)
        auc_bal = compute_pr_auc(rf_bal, X_test, y_test)
        print(f"\n--- PR-AUC (threshold-independent ranking quality) ---")
        print(f"  RF default:  {auc_def:.3f}")
        print(f"  RF balanced: {auc_bal:.3f}")
        print(
            "Note: class_weight='balanced' shifts the operating point at a fixed "
            "threshold; it does not improve the underlying ranking (PR-AUC)."
        )

        # Task 5: PR curves + calibration curves
        plot_pr_curves(rf, rf_bal, X_test, y_test, "results/pr_curves.png")
        plot_calibration_curves(
            rf, rf_bal, X_test, y_test, "results/calibration_curves.png"
        )

    # Task 6: Tree-vs-linear disagreement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = build_logistic_regression(X_train_scaled, y_train)
    if rf is not None and lr is not None:
        d = find_tree_vs_linear_disagreement(
            rf, lr, X_test, X_test_scaled, y_test, NUMERIC_FEATURES
        )
        if d:
            print(
                f"\n--- Tree-vs-linear disagreement (sample idx={d['sample_idx']}) ---"
            )
            print(
                f"  RF P(churn=1)={d['rf_proba']:.3f}  LR P(churn=1)={d['lr_proba']:.3f}"
            )
            print(f"  |diff| = {d['prob_diff']:.3f}   true label = {d['true_label']}")
            print(f"  Feature values: {d['feature_values']}")


if __name__ == "__main__":
    main()
