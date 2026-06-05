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
    precision_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import permutation_importance

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


def build_logistic_regression(
    X_train_scaled, y_train, class_weight=None, random_state=42
):
    """Train a LogisticRegression baseline on scaled features.

    Args:
        class_weight: None for default, 'balanced' to reweight the loss
            so minority-class samples count more during training.
        random_state: Random seed.

    Returns:
        Fitted LogisticRegression(max_iter=1000).
    """
    clf = LogisticRegression(
        max_iter=1000, class_weight=class_weight, random_state=random_state
    )
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
# Tier 1 — Threshold tuning
# ---------------------------------------------------------------------------


def sweep_thresholds(model, X_test, y_test, output_path, thresholds=None):
    """Sweep thresholds 0.10–0.90, plot precision/recall/F1, save PNG.

    Args:
        model: Fitted classifier with predict_proba.
        X_test: Test features (raw, unscaled — RF consumes these).
        y_test: True binary labels.
        output_path: Where to save the PNG.
        thresholds: Array of thresholds to sweep (default 0.10–0.90 step 0.05).

    Returns:
        Dict with keys:
          - best_f1_threshold (float)
          - threshold_80_recall (float or None)
          - results (list of dicts with threshold/precision/recall/f1)
    """

    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_true = np.asarray(y_test)

    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)

    best_f1_idx = int(np.argmax(f1s))
    best_f1_threshold = float(thresholds[best_f1_idx])

    # First threshold where recall >= 0.80
    recall_80_mask = recalls >= 0.80
    threshold_80_recall = (
        float(thresholds[recall_80_mask][0]) if recall_80_mask.any() else None
    )

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, precisions, marker="o", label="Precision", color="#2196F3")
    ax.plot(thresholds, recalls, marker="s", label="Recall", color="#4CAF50")
    ax.plot(thresholds, f1s, marker="^", label="F1", color="#FF5722")

    ax.axvline(
        best_f1_threshold,
        color="#FF5722",
        linestyle="--",
        alpha=0.7,
        label=f"Best F1 threshold = {best_f1_threshold:.2f}",
    )
    if threshold_80_recall is not None:
        ax.axvline(
            threshold_80_recall,
            color="#4CAF50",
            linestyle=":",
            alpha=0.7,
            label=f"80% Recall threshold = {threshold_80_recall:.2f}",
        )

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(
        "Threshold Sweep — Balanced RF\n"
        "Precision, Recall, and F1 vs Decision Threshold"
    )
    ax.legend(loc="center left")
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    results = [
        {
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }
        for t, p, r, f in zip(thresholds, precisions, recalls, f1s)
    ]
    return {
        "best_f1_threshold": best_f1_threshold,
        "threshold_80_recall": threshold_80_recall,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Tier 2 — Permutation importance vs MDI
# ---------------------------------------------------------------------------


def plot_permutation_vs_mdi(rf_balanced, X_test, y_test, feature_names, output_path):
    """Side-by-side bar chart: MDI vs permutation importance.

    Args:
        rf_balanced: Fitted RandomForestClassifier (balanced).
        X_test: Raw test features (DataFrame or array).
        y_test: True labels.
        feature_names: List of feature name strings.
        output_path: Where to save the PNG.

    Returns:
        Dict with keys 'mdi' and 'permutation', each a dict of
        {feature_name: importance} sorted descending.
    """

    # MDI importances (already computed on train set internally by sklearn)
    mdi = dict(zip(feature_names, rf_balanced.feature_importances_))

    # Permutation importance on the held-out test set (more reliable)
    perm_result = permutation_importance(
        rf_balanced,
        X_test,
        y_test,
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
        scoring="average_precision",
    )
    perm = dict(zip(feature_names, perm_result.importances_mean))

    # Sort both by MDI descending for consistent ordering
    features_sorted = sorted(feature_names, key=lambda f: mdi[f], reverse=True)

    mdi_vals = [mdi[f] for f in features_sorted]
    perm_vals = [perm[f] for f in features_sorted]

    # ── Plot ──────────────────────────────────────────────────────────────
    x = np.arange(len(features_sorted))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_mdi = ax.bar(
        x - width / 2, mdi_vals, width, label="MDI (train)", color="#2196F3", alpha=0.85
    )
    bars_perm = ax.bar(
        x + width / 2,
        perm_vals,
        width,
        label="Permutation (test)",
        color="#FF5722",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(features_sorted, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Importance")
    ax.set_title(
        "Feature Importance: MDI vs Permutation\n"
        "Balanced Random Forest — Top 8 Features"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "mdi": {f: mdi[f] for f in features_sorted},
        "permutation": {f: perm[f] for f in features_sorted},
    }


# ---------------------------------------------------------------------------
# Tier 3 — Custom voting ensemble
# ---------------------------------------------------------------------------


class VotingEnsemble:
    """Custom soft-voting ensemble over a list of fitted sklearn classifiers.

    Implements predict() via majority hard vote and predict_proba() via
    averaged probabilities. Handles classifiers whose .classes_ attribute
    may not be ordered [0, 1] by remapping columns before averaging.

    Args:
        estimators: List of (name, fitted_clf) tuples, sklearn convention.
    """

    def __init__(self, estimators):
        self.estimators = estimators  # [(name, clf), ...]
        self.classes_ = np.array([0, 1])  # binary assumption

    # VotingEnsemble does not need .fit() because all estimators are already
    # fitted before being passed in — but we expose a no-op for sklearn
    # convention compatibility.
    def fit(self, X, y):
        return self

    def _align_proba(self, clf, X):
        """Return predict_proba columns guaranteed in [class-0, class-1] order."""
        proba = clf.predict_proba(X)
        classes = list(clf.classes_)
        if classes == [0, 1]:
            return proba
        # Re-order columns to match [0, 1]
        idx0 = classes.index(0)
        idx1 = classes.index(1)
        return proba[:, [idx0, idx1]]

    def predict_proba(self, X):
        """Average predict_proba across all estimators (soft voting)."""
        probas = [self._align_proba(clf, X) for _, clf in self.estimators]
        return np.mean(probas, axis=0)

    def predict(self, X):
        """Majority hard vote across all estimators."""
        votes = np.stack(
            [clf.predict(X) for _, clf in self.estimators], axis=1
        )  # shape (n_samples, n_estimators)
        # Majority vote: sum > half the classifiers → predict 1
        return (votes.sum(axis=1) > (len(self.estimators) / 2)).astype(int)


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
        plt.figure(figsize=(14, 8))
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
    lr_bal = build_logistic_regression(X_train_scaled, y_train, class_weight="balanced")

    if lr is not None and lr_bal is not None:
        print(f"\n--- Logistic Regression (default, class_weight=None) ---")
        print(classification_report(y_test, lr.predict(X_test_scaled), zero_division=0))

        print(f"--- Logistic Regression (balanced, class_weight='balanced') ---")
        print(
            classification_report(
                y_test, lr_bal.predict(X_test_scaled), zero_division=0
            )
        )

        r_lr_def = evaluate_recall_at_threshold(lr, X_test_scaled, y_test)
        r_lr_bal = evaluate_recall_at_threshold(lr_bal, X_test_scaled, y_test)
        auc_lr_def = compute_pr_auc(lr, X_test_scaled, y_test)
        auc_lr_bal = compute_pr_auc(lr_bal, X_test_scaled, y_test)

        print(f"  LR default  recall@0.5: {r_lr_def:.3f}   PR-AUC: {auc_lr_def:.3f}")
        print(f"  LR balanced recall@0.5: {r_lr_bal:.3f}   PR-AUC: {auc_lr_bal:.3f}")
        print(
            "Note: same pattern as RF — class_weight shifts recall at the default "
            "0.5 threshold but does not improve PR-AUC ranking."
        )

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

    # ── Tier 1: Threshold sweep ───────────────────────────────────────────
    if rf_bal is not None:
        print("\n=== Tier 1 — Threshold Sweep (Balanced RF) ===")
        sweep = sweep_thresholds(rf_bal, X_test, y_test, "results/threshold_sweep.png")
        print(f"  Best F1 threshold      : {sweep['best_f1_threshold']:.2f}")
        print(f"  >=80% Recall threshold : {sweep['threshold_80_recall']}")
        print("\n  Threshold  Precision  Recall    F1")
        print("  " + "-" * 42)
        for row in sweep["results"]:
            print(
                f"  {row['threshold']:.2f}       "
                f"{row['precision']:.3f}     "
                f"{row['recall']:.3f}     "
                f"{row['f1']:.3f}"
            )
        t80 = sweep["threshold_80_recall"]
        print(f"\n  >> Recommendation for 200 contacts/month:")
        if t80 is not None:
            print(f"     Use threshold={t80:.2f} to achieve >=80% recall.")
        print("     Lower thresholds catch more churners (fewer false negatives)")
        print("     but waste retention budget on non-churners (more false positives).")
        print("     The F1-maximising threshold balances both costs optimally.")

    # ── Tier 2: Permutation vs MDI ────────────────────────────────────────
    if rf_bal is not None:
        print("\n=== Tier 2 — Permutation vs MDI Importance ===")
        imp_compare = plot_permutation_vs_mdi(
            rf_bal, X_test, y_test, NUMERIC_FEATURES, "results/permutation_vs_mdi.png"
        )
        print(f"  {'Feature':<22s}  {'MDI':>8s}  {'Permutation':>12s}")
        print("  " + "-" * 46)
        for feat in imp_compare["mdi"]:
            mdi_v = imp_compare["mdi"][feat]
            perm_v = imp_compare["permutation"][feat]
            print(f"  {feat:<22s}  {mdi_v:>8.3f}  {perm_v:>12.4f}")
        """Explaination: 
              Permutation importance and MDI (Mean Decrease in Impurity 
              can produce different feature rankings because they measure 
              importance in fundamentally different ways. MDI is calculated 
              during training by tracking how much each feature reduces 
              impurity (e.g., Gini) across all trees. However, this method 
              is biased toward high-cardinality features, such as continuous 
              variables or ID-like columns, because these features offer more 
              potential split points and are therefore more likely to appear 
              important—even if they are not truly predictive.

              In contrast, permutation importance is computed after training by randomly
              shuffling the values of a single feature and measuring the drop in model performance. 
              This approach directly evaluates how much the model depends on that feature for making 
              predictions. As a result, permutation importance provides a more realistic estimate of feature usefulness.

              The two methods tend to disagree when a feature appears structurally useful for splitting
               (high MDI) but does not meaningfully contribute to prediction accuracy (low permutation importance).
               This often happens with noisy or high-cardinality features. Conversely, features that are highly predictive 
              but less frequently used in splits may have lower MDI but higher permutation importance. Therefore, permutation
             importance is generally more reliable for interpreting model behavior, especially when dealing with complex 
              or high-dimensional data."""
    # ── Tier 3: Custom voting ensemble ────────────────────────────────────
    if rf_bal is not None and lr_bal is not None:
        print("\n=== Tier 3 — Custom Voting Ensemble ===")

        dt_bal = build_decision_tree(X_train, y_train, max_depth=5)

        class ScaledLR:
            """Wraps a fitted LR + its scaler into one sklearn-compatible object."""

            def __init__(self, lr_model, fitted_scaler):
                self.lr = lr_model
                self.scaler = fitted_scaler
                self.classes_ = lr_model.classes_

            def predict_proba(self, X):
                return self.lr.predict_proba(self.scaler.transform(X))

            def predict(self, X):
                return self.lr.predict(self.scaler.transform(X))

        slr_bal = ScaledLR(lr_bal, scaler)

        ensemble = VotingEnsemble(
            [
                ("lr_balanced", slr_bal),
                ("dt_balanced", dt_bal),
                ("rf_balanced", rf_bal),
            ]
        )

        print("\n  -- Ensemble (LR_bal + DT_bal + RF_bal) --")
        print(classification_report(y_test, ensemble.predict(X_test), zero_division=0))

        auc_ensemble = average_precision_score(
            y_test, ensemble.predict_proba(X_test)[:, 1]
        )
        auc_rf_bal = compute_pr_auc(rf_bal, X_test, y_test)
        auc_dt_bal = compute_pr_auc(dt_bal, X_test, y_test)
        auc_slr_bal = compute_pr_auc(slr_bal, X_test, y_test)

        print(f"  PR-AUC comparison:")
        print(f"    LR balanced  : {auc_slr_bal:.3f}")
        print(f"    DT balanced  : {auc_dt_bal:.3f}")
        print(f"    RF balanced  : {auc_rf_bal:.3f}")
        print(f"    Ensemble     : {auc_ensemble:.3f}")

        best_individual = max(auc_slr_bal, auc_dt_bal, auc_rf_bal)
        if auc_ensemble > best_individual:
            print("  Ensemble OUTPERFORMS the best individual model on PR-AUC.")
        else:
            print("  Ensemble does NOT outperform the best individual model.")
        print(
            "  Note: ensembles help most when constituent models make different errors."
        )
        print(
            "  When one model dominates (RF >> LR, DT), averaging dilutes its signal."
        )


if __name__ == "__main__":
    main()
