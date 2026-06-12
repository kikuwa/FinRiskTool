"""PU Bagging 训练执行（供 HTTP 与 autoresearch 共用）。"""
import os
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from app.services.data_core.pu.bagging import (
    PUTrainStoppedError,
    PUTrainTimeoutError,
    PU_RUN_TIMEOUT_SECONDS,
    run_pu_learning_pipeline_with_timeout,
)
from app.services.data_core.shared.data_loader import DataLoader


def _check_stop_event(stop_event: Any) -> None:
    if stop_event is not None and stop_event.is_set():
        raise PUTrainStoppedError('用户已停止 PU Bagging 训练')


def load_and_preprocess_pu_data(
    project_root: str,
    label_col: str = 'label',
    dataset_type: str = 'split',
    stop_event: Any = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    full_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """在训练子进程内加载并预处理数据（便于停止时 taskkill 整个子进程）。"""
    from app.services.data_core.shared.dataset_paths import (
        resolve_full_dataset_path,
        resolve_split_paths,
    )

    _check_stop_event(stop_event)
    loader = DataLoader(label_col=label_col)

    if dataset_type == 'full':
        file_path = resolve_full_dataset_path(project_root, full_path=full_path)
        train_df, test_df = loader.load_full_dataset(file_path)
    else:
        train_p, test_p = resolve_split_paths(
            project_root, train_path=train_path, test_path=test_path,
        )
        train_df, test_df = loader.load_train_test_split(train_p, test_p)

    _check_stop_event(stop_event)
    train_df = loader.preprocess_data(train_df, fit_encoders=True)
    _check_stop_event(stop_event)
    test_df = loader.preprocess_data(test_df, fit_encoders=False)
    return train_df, test_df


def _load_csv_robust_local(path: str):
    import pandas as pd
    for enc in ('utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding='utf-8', errors='replace')


def execute_pu_model_training(
    project_root: str,
    label_col: str = 'label',
    dataset_type: str = 'split',
    estimated_positive_rate: float = 0.1,
    pu_params: Dict[str, Any] = None,
    num_boost_round: int = 1200,
    timeout_seconds: int = None,
    save_prediction_files: bool = True,
    stop_checker: Optional[Callable[[], bool]] = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    full_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    加载数据、运行 PU 流水线并返回与 /run_model 一致的结果结构。
  超时仍会写入 pu_train.tsv 并抛出 PUTrainTimeoutError。
    """
    if stop_checker and stop_checker():
        raise PUTrainStoppedError('用户已停止 PU Bagging 训练')

    pu_params = dict(pu_params or {})
    pu_params.setdefault('num_boost_round', num_boost_round)
    if timeout_seconds is None:
        timeout_seconds = PU_RUN_TIMEOUT_SECONDS
    output_dir = os.path.join(project_root, 'data', 'results', 'pu_learning')

    pipeline_result = run_pu_learning_pipeline_with_timeout(
        project_root=project_root,
        label_col=label_col,
        dataset_type=dataset_type,
        train_path=train_path,
        test_path=test_path,
        full_path=full_path,
        output_dir=output_dir,
        estimated_positive_rate=estimated_positive_rate,
        timeout_seconds=timeout_seconds,
        stop_checker=stop_checker,
        save_prediction_files=save_prediction_files,
        **pu_params,
    )

    predictions_path = pipeline_result['predictions_path']
    df = _load_csv_robust_local(predictions_path)

    top_10 = df.nlargest(10, 'prob')[['prob']]
    positive_samples = df[df[label_col] == 1]
    min_positive_confidence = (
        positive_samples['prob'].min() if not positive_samples.empty else 0
    )
    high_confidence_count = len(df[df['prob'] >= 0.9])
    total_samples = len(df)

    return {
        'success': True,
        'test_evaluation': pipeline_result['test_evaluation'],
        'train_predictions_path': pipeline_result['train_predictions_path'],
        'pu_train_log_path': pipeline_result.get('pu_train_log_path'),
        'run_time_seconds': pipeline_result.get('run_time_seconds'),
        'top_10': top_10.reset_index().to_dict('records'),
        'min_positive_confidence': min_positive_confidence,
        'high_confidence_count': high_confidence_count,
        'total_samples': total_samples,
    }


def load_pu_results_from_disk(project_root: str) -> Optional[Dict[str, Any]]:
    """从磁盘恢复与 /run_model 一致的结果摘要（用于页面 init）。"""
    from app.services.data_core.pu.bagging import (
        compute_positive_metrics_at_threshold,
        threshold_from_estimated_positive_rate,
    )
    from app.services.data_core.shared.pu_session import read_pu_session, resolve_label_col

    output_dir = os.path.join(project_root, 'data', 'results', 'pu_learning')
    test_path = os.path.join(output_dir, 'test_predictions.csv')
    if not os.path.isfile(test_path):
        return None

    label_col = resolve_label_col(project_root)
    df = _load_csv_robust_local(test_path)
    if 'prob' not in df.columns:
        return None

    session = read_pu_session(project_root) or {}
    rate = session.get('estimated_positive_rate', 0.1)
    if rate is None:
        rate = 0.1
    elif float(rate) > 1:
        rate = float(rate) / 100.0
    else:
        rate = float(rate)

    test_evaluation = None
    if label_col in df.columns:
        y_test = df[label_col]
        test_probs = df['prob'].values
        threshold, predicted_positive_count = threshold_from_estimated_positive_rate(
            test_probs, rate
        )
        metrics = compute_positive_metrics_at_threshold(
            y_test.values, test_probs, threshold
        )
        test_evaluation = {
            'estimated_positive_rate': rate,
            'predicted_positive_count': predicted_positive_count,
            'test_sample_count': len(test_probs),
            'test_positive_count': int((y_test == 1).sum()),
            **metrics,
        }

    top_10 = df.nlargest(10, 'prob')[['prob']]
    if label_col in df.columns:
        positive_samples = df[df[label_col] == 1]
        min_positive_confidence = (
            positive_samples['prob'].min() if not positive_samples.empty else 0
        )
    else:
        min_positive_confidence = 0

    high_confidence_count = len(df[df['prob'] >= 0.9])
    return {
        'success': True,
        'restored_from_disk': True,
        'test_evaluation': test_evaluation,
        'top_10': top_10.reset_index().to_dict('records'),
        'min_positive_confidence': float(min_positive_confidence),
        'high_confidence_count': high_confidence_count,
        'total_samples': len(df),
    }
