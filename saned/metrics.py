"""Sentence-level predictions and evaluation metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score)

def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    """计算相关系数；输入过短或为常量时返回 NaN。"""
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float((pearsonr if method == "pearson" else spearmanr)(y_true, y_pred).statistic)


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    """返回一个回归目标的误差和相关性指标。"""
    yt, yp = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    mean_true, mean_pred = float(yt.mean()), float(yp.mean())
    covariance = float(np.mean((yt - mean_true) * (yp - mean_pred)))
    ccc_denominator = float(yt.var() + yp.var() + (mean_true - mean_pred) ** 2)
    return {
        "mse": float(mean_squared_error(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "pearson": safe_corr(yt, yp, "pearson"),
        "spearman": safe_corr(yt, yp, "spearman"),
        "ccc": (2.0 * covariance / ccc_denominator) if ccc_denominator > 1e-12 else float("nan"),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else float("nan"),
    }


def classification_metrics(
    y_true: list[int], y_pred: list[int], y_prob: np.ndarray | None = None
) -> dict:
    """评估有效的三分类目标，并忽略哨兵标签。"""
    mask = np.asarray(y_true) >= 0
    yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    if len(yt) == 0:
        return {"n": 0, "accuracy": float("nan"), "balanced_accuracy": float("nan"), "macro_f1": float("nan")}
    labels = [0, 1, 2]
    precision, recall, per_f1, support = precision_recall_fscore_support(
        yt, yp, labels=labels, zero_division=0
    )
    result = {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(per_f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist(),
    }
    if y_prob is not None:
        prob = np.asarray(y_prob, dtype=np.float64)[mask]
        confidence = prob.max(axis=1)
        correct = yp == yt
        ece = 0.0
        for lower in np.linspace(0.0, 0.9, 10):
            in_bin = (confidence > lower) & (confidence <= lower + 0.1)
            if np.any(in_bin):
                ece += float(in_bin.mean()) * abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean()))
        one_hot = np.eye(3, dtype=np.float64)[yt]
        result["expected_calibration_error"] = float(ece)
        result["brier_score"] = float(np.square(prob - one_hot).sum(axis=1).mean())
    return result


def prediction_metrics(predictions: pd.DataFrame) -> tuple[dict, dict, dict]:
    """从统一预测表计算分类与两项回归指标。"""
    probability_columns = ["class_prob_0", "class_prob_1", "class_prob_2"]
    probabilities = (
        predictions[probability_columns].to_numpy()
        if set(probability_columns) <= set(predictions.columns)
        else None
    )
    classification = classification_metrics(
        predictions.class_true, predictions.class_pred, probabilities
    )
    valence = regression_metrics(predictions.valence_true, predictions.valence_pred)
    arousal = regression_metrics(predictions.arousal_true, predictions.arousal_pred)
    return classification, valence, arousal


@torch.no_grad()
def evaluate(model, loader, device):
    """执行推理并返回样本级目标值与预测值。"""
    model.eval()
    out = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        pred = model(x)
        class_pred = pred["class_logits"].argmax(dim=-1).cpu().numpy()
        class_prob = torch.softmax(pred["class_logits"], dim=-1).cpu().numpy()
        for i, subject in enumerate(batch["subject"]):
            out.append({
                "subject_id": subject,
                "center": int(batch["center"][i]),
                "event_id": str(batch["segment_id"][i]),
                "segment_id": str(batch["segment_id"][i]),
                "class_true": int(batch["class"][i]),
                "class_pred": int(class_pred[i]),
                "class_prob_0": float(class_prob[i, 0]),
                "class_prob_1": float(class_prob[i, 1]),
                "class_prob_2": float(class_prob[i, 2]),
                "valence_true": float(batch["valence"][i]),
                "valence_pred": float(pred["valence"][i].cpu()),
                "arousal_true": float(batch["arousal"][i]),
                "arousal_pred": float(pred["arousal"][i].cpu()),
            })
    return pd.DataFrame(out)


def aggregate_sentence_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """按 subject/sentence 聚合预测，作为主句子级评价单位。"""
    if predictions.empty:
        return predictions.copy()
    rows = []
    group_cols = ["subject_id", "segment_id"]
    for (subject, segment_id), group in predictions.groupby(group_cols, sort=False):
        probs = group[["class_prob_0", "class_prob_1", "class_prob_2"]].to_numpy().mean(axis=0)
        rows.append(
            {
                "subject_id": subject,
                "event_id": str(segment_id),
                "segment_id": str(segment_id),
                "center": int(group.center.iloc[0]),
                "class_true": int(group.class_true.iloc[0]),
                "class_pred": int(np.argmax(probs)),
                "class_prob_0": float(probs[0]),
                "class_prob_1": float(probs[1]),
                "class_prob_2": float(probs[2]),
                "valence_true": float(group.valence_true.iloc[0]),
                "valence_pred": float(group.valence_pred.mean()),
                "arousal_true": float(group.arousal_true.iloc[0]),
                "arousal_pred": float(group.arousal_pred.mean()),
                "n_context_views": int(len(group)),
            }
        )
    return pd.DataFrame(rows)

