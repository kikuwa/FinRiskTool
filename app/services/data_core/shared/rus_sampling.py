"""RandomUnderSampler 策略：按少数/多数比构造 dict，避免 imblearn 不可达比例报错。"""
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

DEFAULT_RUS_RATIO = 0.07
DEFAULT_RUS_MIN_EFFECTIVE = 0.12


def rus_dict_sampling_strategy(
    y: Union[pd.Series, np.ndarray],
    desired_minority_majority_ratio: float = DEFAULT_RUS_RATIO,
    min_effective_ratio: float = DEFAULT_RUS_MIN_EFFECTIVE,
) -> Tuple[Dict[int, int], float]:
    """
    仅欠采样多数类。desired = n_minority / n_majority（重采样后）。
    若 desired 低于数据天然比例，则提升到 min_effective_ratio（或略高于天然比）。
    """
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        return {int(labels[0]): int(counts[0])}, 1.0

    ordered = sorted(zip(labels, counts), key=lambda x: x[1])
    minor_label, n_minor = int(ordered[0][0]), int(ordered[0][1])
    major_label, n_major = int(ordered[1][0]), int(ordered[1][1])

    achievable = n_minor / n_major if n_major else 1.0
    ratio = max(float(desired_minority_majority_ratio), 1e-6)
    if ratio < achievable - 1e-9:
        ratio = max(min_effective_ratio, achievable + 0.005)

    n_major_target = min(n_major, max(1, int(np.ceil(n_minor / ratio))))
    strategy = {major_label: n_major_target, minor_label: n_minor}
    effective = n_minor / n_major_target if n_major_target else ratio
    return strategy, effective
