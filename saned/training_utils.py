"""Deterministic subject splits and per-update cosine scheduling."""
from __future__ import annotations
import math
import random
import numpy as np
import torch
from sklearn.model_selection import KFold
from .config import Config

def seed_everything(seed: int) -> None:
    """为 Python、NumPy 和 PyTorch 设置随机种子，保证分折训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """从初始学习率直接开始，并按更新步余弦衰减到最小学习率。"""
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    min_lr_ratio = cfg.min_learning_rate / cfg.learning_rate

    def lr_multiplier(update_step: int) -> float:
        """返回某次优化器更新对应的余弦衰减比例。"""
        if total_steps <= 1:
            return 1.0 if update_step == 0 else min_lr_ratio
        progress = min(1.0, update_step / (total_steps - 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def make_cv_splits(
    subjects: list[str], n_splits: int, n_val_subjects: int, seed: int
) -> list[dict]:
    """创建确定性的被试级分折，并保证验证被试彼此独立。"""
    if not 2 <= n_splits <= len(subjects):
        raise ValueError(f"cv_folds must be between 2 and {len(subjects)}, got {n_splits}")
    if not 1 <= n_val_subjects < len(subjects) - math.ceil(len(subjects) / n_splits):
        raise ValueError("val_subjects_per_fold leaves no subjects for model fitting")
    outer = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    subject_array = np.asarray(subjects)
    for fold_index, (outer_train_idx, test_idx) in enumerate(outer.split(subject_array), start=1):
        outer_train = subject_array[outer_train_idx].tolist()
        test_subjects = subject_array[test_idx].tolist()
        rng = random.Random(seed + fold_index)
        val_subjects = sorted(rng.sample(outer_train, n_val_subjects))
        fit_subjects = [s for s in outer_train if s not in val_subjects]
        splits.append(
            {
                "fold": fold_index,
                "fit_subjects": fit_subjects,
                "validation_subjects": val_subjects,
                "test_subjects": test_subjects,
            }
        )
    return splits

