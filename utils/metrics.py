"""ML evaluation metrics implemented from scratch with NumPy.

All implementations match sklearn's behavior for standard inputs.
"""
import numpy as np
from typing import Optional, List


# ─────────────────────────── Classification Metrics ──────────────────

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples.

    accuracy = correct / total

    Example:
        accuracy_score([0, 1, 1, 0], [0, 1, 0, 0]) → 0.75
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def _confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1):
    """Compute TP, FP, TN, FN for binary classification."""
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    tn = np.sum((y_pred != pos_label) & (y_true != pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    return int(tp), int(fp), int(tn), int(fn)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Confusion matrix for multi-class classification.

    Returns:
        Matrix C where C[i,j] = number of samples with true label i
        and predicted label j.

    Example:
        confusion_matrix([0,1,2,0,1,2], [0,2,2,0,0,2])
        → [[2,0,0], [1,0,1], [0,0,2]]
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    pos_label=1,
) -> float:
    """Precision = TP / (TP + FP).

    How many predicted positives are actually positive?

    Args:
        average: "binary", "macro", "micro", or "weighted"
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if average == "binary":
        tp, fp, _, _ = _confusion_matrix_binary(y_true, y_pred, pos_label)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    labels = np.unique(y_true)
    precisions, supports = [], []
    for label in labels:
        tp, fp, _, _ = _confusion_matrix_binary(y_true, y_pred, label)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(p)
        supports.append(np.sum(y_true == label))

    if average == "macro":
        return float(np.mean(precisions))
    elif average == "micro":
        total_tp = sum(
            _confusion_matrix_binary(y_true, y_pred, l)[0] for l in labels
        )
        total_fp = sum(
            _confusion_matrix_binary(y_true, y_pred, l)[1] for l in labels
        )
        return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    elif average == "weighted":
        total = sum(supports)
        return float(sum(p * s for p, s in zip(precisions, supports)) / total) if total > 0 else 0.0
    raise ValueError(f"Unknown average: {average}")


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    pos_label=1,
) -> float:
    """Recall (Sensitivity) = TP / (TP + FN).

    How many actual positives were correctly identified?
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if average == "binary":
        tp, _, _, fn = _confusion_matrix_binary(y_true, y_pred, pos_label)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    labels = np.unique(y_true)
    recalls, supports = [], []
    for label in labels:
        tp, _, _, fn = _confusion_matrix_binary(y_true, y_pred, label)
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(r)
        supports.append(np.sum(y_true == label))

    if average == "macro":
        return float(np.mean(recalls))
    elif average == "micro":
        total_tp = sum(_confusion_matrix_binary(y_true, y_pred, l)[0] for l in labels)
        total_fn = sum(_confusion_matrix_binary(y_true, y_pred, l)[3] for l in labels)
        return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    elif average == "weighted":
        total = sum(supports)
        return float(sum(r * s for r, s in zip(recalls, supports)) / total) if total > 0 else 0.0
    raise ValueError(f"Unknown average: {average}")


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    pos_label=1,
) -> float:
    """F1 = harmonic mean of precision and recall.

    F1 = 2 * P * R / (P + R)

    Balances precision and recall — useful for imbalanced classes.
    """
    p = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    r = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area under the ROC curve (binary classification).

    Args:
        y_true: Binary labels {0, 1}
        y_scores: Predicted probabilities for class 1

    Returns:
        AUC in [0, 1]. 0.5 = random, 1.0 = perfect.
    """
    y_true, y_scores = np.asarray(y_true), np.asarray(y_scores)
    # Equivalent to: fraction of (positive, negative) pairs where
    # positive has higher score (Mann-Whitney U statistic)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    # Count pairs where positive score > negative score
    correct_pairs = sum(
        np.sum(pos_score > neg_scores) + 0.5 * np.sum(pos_score == neg_scores)
        for pos_score in pos_scores
    )
    return float(correct_pairs / (n_pos * n_neg))


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None
) -> str:
    """Print a text classification report like sklearn's."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = sorted(np.unique(y_true))

    lines = []
    lines.append(f"\n{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    lines.append("")

    total_support = 0
    for label in labels:
        p = precision_score(y_true, y_pred, average="binary", pos_label=label)
        r = recall_score(y_true, y_pred, average="binary", pos_label=label)
        f = f1_score(y_true, y_pred, average="binary", pos_label=label)
        s = int(np.sum(y_true == label))
        total_support += s
        lines.append(f"{str(label):>20} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>10}")

    lines.append("")
    acc = accuracy_score(y_true, y_pred)
    lines.append(f"{'accuracy':>20} {'':>10} {'':>10} {acc:>10.4f} {total_support:>10}")
    p_macro = precision_score(y_true, y_pred, average="macro")
    r_macro = recall_score(y_true, y_pred, average="macro")
    f_macro = f1_score(y_true, y_pred, average="macro")
    lines.append(f"{'macro avg':>20} {p_macro:>10.4f} {r_macro:>10.4f} {f_macro:>10.4f} {total_support:>10}")
    p_w = precision_score(y_true, y_pred, average="weighted")
    r_w = recall_score(y_true, y_pred, average="weighted")
    f_w = f1_score(y_true, y_pred, average="weighted")
    lines.append(f"{'weighted avg':>20} {p_w:>10.4f} {r_w:>10.4f} {f_w:>10.4f} {total_support:>10}")

    return "\n".join(lines)


# ─────────────────────────── Regression Metrics ──────────────────────

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error. Penalizes large errors more.

    MSE = (1/n) * sum((y - y_hat)^2)
    """
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error. Same units as target variable."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error. More robust to outliers than MSE.

    MAE = (1/n) * sum(|y - y_hat|)
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error. Scale-independent.

    MAPE = (100/n) * sum(|y - y_hat| / |y|)
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(100 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot
    R² = 1.0: perfect prediction
    R² = 0.0: baseline (predicting mean)
    R² < 0.0: worse than baseline

    Example:
        r2_score([1,2,3,4], [1.1,2.0,2.9,4.2]) → ~0.99
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """Huber loss — combines MSE and MAE for robustness to outliers.

    L = 0.5*(y-p)^2 if |y-p| <= delta else delta*(|y-p| - 0.5*delta)
    """
    residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    mask = residuals <= delta
    loss = np.where(mask, 0.5 * residuals**2, delta * (residuals - 0.5 * delta))
    return float(np.mean(loss))


if __name__ == "__main__":
    import sklearn.metrics as sk_metrics

    np.random.seed(42)

    # ── Classification ─────────────────────────────────────────────────
    print("=" * 60)
    print("Classification Metrics Verification")
    print("=" * 60)
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 1])

    metrics = [
        ("accuracy", accuracy_score(y_true, y_pred), sk_metrics.accuracy_score(y_true, y_pred)),
        ("precision", precision_score(y_true, y_pred), sk_metrics.precision_score(y_true, y_pred)),
        ("recall", recall_score(y_true, y_pred), sk_metrics.recall_score(y_true, y_pred)),
        ("f1", f1_score(y_true, y_pred), sk_metrics.f1_score(y_true, y_pred)),
    ]

    print(f"\n{'Metric':<15} {'Ours':>10} {'sklearn':>10} {'Match':>8}")
    print("-" * 45)
    for name, ours, sk in metrics:
        match = "✓" if abs(ours - sk) < 1e-6 else "✗"
        print(f"{name:<15} {ours:>10.4f} {sk:>10.4f} {match:>8}")

    print(classification_report(y_true, y_pred))

    # ── Regression ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Regression Metrics Verification")
    print("=" * 60)
    y_r_true = np.array([3.0, 2.5, 4.0, 1.5, 5.0, 2.0])
    y_r_pred = np.array([2.8, 2.7, 3.9, 1.8, 4.7, 2.3])

    reg_metrics = [
        ("MSE", mse(y_r_true, y_r_pred), sk_metrics.mean_squared_error(y_r_true, y_r_pred)),
        ("MAE", mae(y_r_true, y_r_pred), sk_metrics.mean_absolute_error(y_r_true, y_r_pred)),
        ("R²", r2_score(y_r_true, y_r_pred), sk_metrics.r2_score(y_r_true, y_r_pred)),
        ("RMSE", rmse(y_r_true, y_r_pred), np.sqrt(sk_metrics.mean_squared_error(y_r_true, y_r_pred))),
    ]

    print(f"\n{'Metric':<15} {'Ours':>10} {'sklearn':>10} {'Match':>8}")
    print("-" * 45)
    for name, ours, sk in reg_metrics:
        match = "✓" if abs(ours - sk) < 1e-6 else "✗"
        print(f"{name:<15} {ours:>10.4f} {sk:>10.4f} {match:>8}")
