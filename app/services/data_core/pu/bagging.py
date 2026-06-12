import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

PU_TRAIN_LOG_FILENAME = 'pu_train.tsv'
PU_TRAIN_LOG_HEADER = [
    '预估正样本比例', '推导阈值', '召回率', '精准率', 'F1', '时间', '参数',
]
PU_TRAIN_PARAM_KEYS = [
    'n_estimators',
    'imbalance_ratio',
    'verbosity',
    'learning_rate',
    'num_leaves',
    'n_jobs',
    'scale_pos_weight',
    'max_depth',
    'min_child_samples',
    'subsample',
    'colsample_bytree',
    'num_boost_round',
]
PU_RUN_TIMEOUT_SECONDS = 600  # 10 分钟
AUTORESEARCH_TMP_TRAIN_PREDICTIONS = '_autoresearch_tmp_train_predictions.csv'
AUTORESEARCH_TMP_TEST_PREDICTIONS = '_autoresearch_tmp_test_predictions.csv'
AUTORESEARCH_BEST_TRAIN_PREDICTIONS = 'best_train_predictions.csv'
AUTORESEARCH_BEST_TEST_PREDICTIONS = 'best_test_predictions.csv'

_active_pu_proc: Optional[mp.Process] = None
_active_pu_stop_event: Any = None
_active_pu_pid: Optional[int] = None
_active_pu_proc_lock = threading.Lock()
_pu_user_stop_pending = False
PU_STOP_POLL_INTERVAL_SECONDS = 0.5


class PUTrainTimeoutError(TimeoutError):
    """PU Bagging 训练超过允许时长。"""


class PUTrainStoppedError(InterruptedError):
    """PU Bagging 训练被用户主动停止。"""


def mark_pu_user_stop_requested() -> None:
    """标记用户已请求停止（子进程尚未启动时 terminate 也会生效）。"""
    global _pu_user_stop_pending
    _pu_user_stop_pending = True


def _user_stop_requested(stop_checker: Optional[Callable[[], bool]] = None) -> bool:
    if _pu_user_stop_pending:
        return True
    return bool(stop_checker and stop_checker())


def _signal_stop_to_worker() -> None:
    """通知训练子进程协作退出（每个子模型迭代之间会检查）。"""
    with _active_pu_proc_lock:
        stop_event = _active_pu_stop_event
    if stop_event is not None:
        stop_event.set()


def _terminate_process_tree(proc: mp.Process) -> None:
    """终止子进程及其子进程树（Windows 上 LightGBM n_jobs>1 会拉起额外进程）。"""
    pid = proc.pid
    if pid is None:
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return

    if sys.platform == 'win32':
        subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(pid)],
            capture_output=True,
            check=False,
        )
    else:
        proc.terminate()

    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join()


def _kill_active_pu_process() -> bool:
    """强制终止已注册的 PU 训练子进程（与超时相同 taskkill 路径）。"""
    with _active_pu_proc_lock:
        proc = _active_pu_proc
        pid = _active_pu_pid
        stop_event = _active_pu_stop_event

    if stop_event is not None:
        stop_event.set()

    killed = False
    if proc is not None:
        try:
            if proc.is_alive():
                print('[PU Bagging] 收到停止信号，正在终止训练子进程…')
                _terminate_process_tree(proc)
                print('[PU Bagging] 训练子进程已终止')
                killed = True
        except (ValueError, OSError):
            pass

    if not killed and pid is not None and sys.platform == 'win32':
        print(f'[PU Bagging] 按 PID {pid} 终止训练进程树…')
        subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(pid)],
            capture_output=True,
            check=False,
        )
        killed = True

    return killed


def _start_pu_stop_watchdog(
    stop_checker: Optional[Callable[[], bool]],
    stop_event: Any,
) -> tuple:
    """独立看门狗线程：HTTP 停止与轮询解耦，避免主线程阻塞在 queue.get 时无法 kill。"""
    cancel = threading.Event()

    def _watch() -> None:
        while not cancel.is_set():
            if _user_stop_requested(stop_checker):
                print('[PU Bagging] 看门狗：检测到用户停止请求，终止训练…')
                _kill_active_pu_process()
                return
            cancel.wait(PU_STOP_POLL_INTERVAL_SECONDS)

    thread = threading.Thread(
        target=_watch,
        name='pu-train-stop-watchdog',
        daemon=True,
    )
    thread.start()
    return cancel, thread


def terminate_active_pu_training() -> bool:
    """终止正在运行的 PU 训练子进程（autoresearch 停止时调用）。"""
    mark_pu_user_stop_requested()
    return _kill_active_pu_process() or _pu_user_stop_pending


def _check_stop_event(stop_event: Any) -> None:
    if stop_event is not None and stop_event.is_set():
        raise PUTrainStoppedError('用户已停止 PU Bagging 训练')


def promote_autoresearch_predictions(output_dir: str) -> None:
    """将 autoresearch 临时预测文件晋升为正式 predictions CSV。"""
    pairs = [
        (AUTORESEARCH_TMP_TRAIN_PREDICTIONS, 'train_predictions.csv'),
        (AUTORESEARCH_TMP_TEST_PREDICTIONS, 'test_predictions.csv'),
    ]
    for tmp_name, final_name in pairs:
        src = os.path.join(output_dir, tmp_name)
        dst = os.path.join(output_dir, final_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'已更新最优预测: {dst}')
    snapshot_best_predictions(output_dir)


def snapshot_best_predictions(output_dir: str) -> None:
    """将当前正式 predictions 复制为最优快照（F1 提升时调用）。"""
    pairs = [
        ('train_predictions.csv', AUTORESEARCH_BEST_TRAIN_PREDICTIONS),
        ('test_predictions.csv', AUTORESEARCH_BEST_TEST_PREDICTIONS),
    ]
    for src_name, dst_name in pairs:
        src = os.path.join(output_dir, src_name)
        dst = os.path.join(output_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)


def finalize_best_predictions(output_dir: str) -> None:
    """autoresearch 结束时，从最优快照恢复正式 predictions（若快照存在）。"""
    pairs = [
        (AUTORESEARCH_BEST_TRAIN_PREDICTIONS, 'train_predictions.csv'),
        (AUTORESEARCH_BEST_TEST_PREDICTIONS, 'test_predictions.csv'),
    ]
    for src_name, dst_name in pairs:
        src = os.path.join(output_dir, src_name)
        dst = os.path.join(output_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'已恢复最优预测: {dst}')


def extract_pu_train_params(kwargs: dict) -> Dict:
    """从流水线入参中提取需写入日志的超参。"""
    return {key: kwargs[key] for key in PU_TRAIN_PARAM_KEYS if key in kwargs}


def format_pu_train_params(params: dict) -> str:
    """序列化超参，使用分号分隔，避免与 TSV 列分隔符冲突。"""
    if not params:
        return 'NAN'
    parts = [f'{key}={params[key]}' for key in PU_TRAIN_PARAM_KEYS if key in params]
    return ';'.join(parts) if parts else 'NAN'


def parse_pu_train_params(params_str: str) -> Dict[str, Any]:
    """解析 pu_train.tsv 参数列（与 format_pu_train_params 对称）。"""
    if not params_str or str(params_str).strip().upper() == 'NAN':
        return {}
    result: Dict[str, Any] = {}
    for part in str(params_str).split(';'):
        part = part.strip()
        if not part or '=' not in part:
            continue
        key, _, raw_val = part.partition('=')
        key = key.strip()
        raw_val = raw_val.strip()
        if key not in PU_TRAIN_PARAM_KEYS:
            continue
        try:
            if '.' in raw_val or 'e' in raw_val.lower():
                result[key] = float(raw_val)
            else:
                result[key] = int(raw_val)
        except ValueError:
            result[key] = raw_val
    return result


def _format_log_value(value: Union[float, str, None]) -> str:
    if value is None:
        return 'NAN'
    if isinstance(value, str):
        return value
    if isinstance(value, float) and np.isnan(value):
        return 'NAN'
    return f'{float(value):.6g}'


def append_pu_train_log(
    output_dir: str,
    estimated_positive_rate: float,
    threshold: Union[float, str, None],
    recall: Union[float, str, None],
    precision: Union[float, str, None],
    f1: Union[float, str, None],
    time_value: str,
    params: Union[dict, str, None] = None,
) -> str:
    """追加一条 PU 训练运行记录到 pu_train.tsv。"""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, PU_TRAIN_LOG_FILENAME)
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

    rate_str = f'{estimated_positive_rate * 100:.4g}%'
    if isinstance(params, dict):
        params_str = format_pu_train_params(params)
    elif params is None:
        params_str = 'NAN'
    else:
        params_str = str(params)

    row = [
        rate_str,
        _format_log_value(threshold),
        _format_log_value(recall),
        _format_log_value(precision),
        _format_log_value(f1),
        time_value,
        params_str,
    ]

    with open(log_path, 'a', encoding='utf-8', newline='') as f:
        if write_header:
            f.write('|'.join(PU_TRAIN_LOG_HEADER) + '\n')
        f.write('|'.join(row) + '\n')

    return log_path


def append_pu_train_log_timeout(
    output_dir: str,
    estimated_positive_rate: float,
    params: Union[dict, None] = None,
) -> str:
    return append_pu_train_log(
        output_dir,
        estimated_positive_rate,
        'NAN',
        'NAN',
        'NAN',
        'NAN',
        'timeout',
        params=params,
    )


def _pu_pipeline_worker(kwargs: dict, result_queue: mp.Queue) -> None:
    from app.utils.session_logging import attach_worker_session_logging

    attach_worker_session_logging()
    started = time.perf_counter()
    stop_event = kwargs.get('stop_event')
    try:
        if 'train_df' not in kwargs or 'test_df' not in kwargs:
            from app.services.data_core.pu.training_runner import load_and_preprocess_pu_data

            project_root = kwargs.pop('project_root')
            dataset_type = kwargs.pop('dataset_type', 'split')
            train_path = kwargs.pop('train_path', None)
            test_path = kwargs.pop('test_path', None)
            full_path = kwargs.pop('full_path', None)
            label_col = kwargs.get('label_col', 'label')
            train_df, test_df = load_and_preprocess_pu_data(
                project_root=project_root,
                label_col=label_col,
                dataset_type=dataset_type,
                stop_event=stop_event,
                train_path=train_path,
                test_path=test_path,
                full_path=full_path,
            )
            kwargs['train_df'] = train_df
            kwargs['test_df'] = test_df

        result = run_pu_learning_pipeline(**kwargs)
        elapsed = time.perf_counter() - started
        result_queue.put({'status': 'ok', 'result': result, 'elapsed': elapsed})
    except PUTrainStoppedError as exc:
        result_queue.put({
            'status': 'stopped',
            'error': str(exc),
            'elapsed': time.perf_counter() - started,
        })
    except Exception as exc:
        result_queue.put({
            'status': 'error',
            'error': str(exc),
            'elapsed': time.perf_counter() - started,
        })


def threshold_from_estimated_positive_rate(
    y_prob: np.ndarray,
    estimated_positive_rate: float,
) -> tuple:
    """
    按预估正样本比例取评分最高的 top-k 样本，其临界分值作为分类阈值。
    例如 3000 条、10% → 第 300 高分样本的 prob 即为阈值。
    返回 (threshold, predicted_positive_count)。
    """
    if not 0 < estimated_positive_rate <= 1:
        raise ValueError('预估正样本比例须在 (0, 1] 之间')

    y_prob = np.asarray(y_prob)
    n = len(y_prob)
    if n == 0:
        raise ValueError('测试集为空，无法计算阈值')

    k = max(1, int(np.ceil(n * estimated_positive_rate)))
    sorted_probs = np.sort(y_prob)[::-1]
    threshold = float(sorted_probs[min(k - 1, n - 1)])
    return threshold, k


def compute_positive_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict:
    """在指定概率阈值下计算正样本召回率、精准率与 F1。"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        'threshold': float(threshold),
        'recall': float(recall),
        'precision': float(precision),
        'f1': float(f1),
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def _dtype_snapshot(df: pd.DataFrame, label_col: str, top_n: int = 20) -> Dict[str, str]:
    """返回前 top_n 个特征列的 dtype 快照，便于日志定位。"""
    cols = [c for c in df.columns if c != label_col][:top_n]
    return {c: str(df[c].dtype) for c in cols}


def _coerce_non_numeric_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
) -> list[str]:
    """
    对非数值特征做兜底转换：
    1) 尝试 to_numeric；
    2) 若无法数值化，则做统一类别编码（train/test 联合映射）。
    返回转换日志列表。
    """
    changed_logs: list[str] = []
    feature_cols = [c for c in train_df.columns if c != label_col]
    for col in feature_cols:
        train_is_num = pd.api.types.is_numeric_dtype(train_df[col])
        test_is_num = pd.api.types.is_numeric_dtype(test_df[col])
        if train_is_num and test_is_num:
            continue

        raw_train = train_df[col]
        raw_test = test_df[col]
        before_train = str(raw_train.dtype)
        before_test = str(raw_test.dtype)

        train_num = pd.to_numeric(raw_train, errors='coerce')
        test_num = pd.to_numeric(raw_test, errors='coerce')
        if train_num.notna().any() or test_num.notna().any():
            fill_val = train_num.dropna().median()
            if pd.isna(fill_val):
                fill_val = 0.0
            train_df[col] = train_num.fillna(fill_val).astype(float)
            test_df[col] = test_num.fillna(fill_val).astype(float)
            changed_logs.append(
                f'{col}: {before_train}/{before_test} -> float64 (to_numeric, fill={fill_val})'
            )
            continue

        # 无法数值化：统一类别编码（train/test 共同词表）
        train_str = raw_train.fillna('Missing').astype(str)
        test_str = raw_test.fillna('Missing').astype(str)
        combined = pd.concat([train_str, test_str], ignore_index=True)
        categories = pd.Index(pd.unique(combined))
        train_df[col] = pd.Categorical(train_str, categories=categories).codes.astype(float)
        test_df[col] = pd.Categorical(test_str, categories=categories).codes.astype(float)
        changed_logs.append(
            f'{col}: {before_train}/{before_test} -> float64 (category_code, cats={len(categories)})'
        )
    return changed_logs


def _ensure_numeric_pu_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """训练前确保特征列全为数值 dtype，并打印前后快照。"""
    print(f'PU dtype snapshot(before train): {_dtype_snapshot(train_df, label_col)}')
    print(f'PU dtype snapshot(before test): {_dtype_snapshot(test_df, label_col)}')

    train_df = train_df.copy()
    test_df = test_df.copy()

    # 标签列兜底为数值
    for name, df in (('train', train_df), ('test', test_df)):
        if not pd.api.types.is_numeric_dtype(df[label_col]):
            coerced = pd.to_numeric(df[label_col], errors='coerce')
            if coerced.isna().any():
                bad = int(coerced.isna().sum())
                raise ValueError(
                    f'{name} 数据标签列 `{label_col}` 无法转为数值，存在 {bad} 个无效值'
                )
            df[label_col] = coerced.astype(int)

    changed_logs = _coerce_non_numeric_features(train_df, test_df, label_col)
    for item in changed_logs:
        print(f'PU dtype normalized: {item}')

    non_numeric = [
        c for c in train_df.columns
        if c != label_col and not pd.api.types.is_numeric_dtype(train_df[c])
    ]
    if non_numeric:
        snapshot = {c: str(train_df[c].dtype) for c in non_numeric[:20]}
        raise ValueError(
            '特征列仍包含非数值类型: '
            + ', '.join(non_numeric[:20])
            + f' | dtype={snapshot}'
        )

    print(f'PU dtype snapshot(after train): {_dtype_snapshot(train_df, label_col)}')
    print(f'PU dtype snapshot(after test): {_dtype_snapshot(test_df, label_col)}')
    return train_df, test_df


class BaggingPULeaning:
    """
    Bagging PU Learning：仅在 fit 阶段使用训练集（P + U），测试集仅用于 predict 与离线评估。
    """

    def __init__(
        self,
        n_estimators=200,
        imbalance_ratio=0.2,
        random_seed=42,
        lgb_params=None,
        num_boost_round=1200,
    ):
        self.n_estimators = n_estimators
        self.imbalance_ratio = imbalance_ratio
        self.random_seed = random_seed
        self.num_boost_round = num_boost_round
        self.lgb_params = lgb_params or {}
        self.models = []
        self.feature_names = []

    def fit(self, X_p, X_u, y_p, y_u, stop_event=None):
        """仅使用训练集正例 P 与未标记 U，不涉及测试集。"""
        self.feature_names = X_p.columns.tolist()
        n_p = len(X_p)
        n_u_sample = int(n_p * self.imbalance_ratio)
        print("开始训练 Bagging PU 模型（共{}个子模型）".format(self.n_estimators))
        print("Positive 样本数: {}, 每次迭代Unlabeled采样数: {}".format(n_p, n_u_sample))

        for i in range(self.n_estimators):
            _check_stop_event(stop_event)
            replace = n_u_sample > len(X_u)
            y_u_subset = y_u.sample(n_u_sample, random_state=self.random_seed + i, replace=replace)
            X_u_subset = X_u.loc[y_u_subset.index]

            X_train = pd.concat([X_p, X_u_subset])
            y_train = pd.concat([y_p, y_u_subset])

            params = {
                'objective': 'binary',
                'metric': 'average_precision',
                'verbosity': -1,
                'learning_rate': 0.05,
                'num_leaves': 20,
                'n_jobs': -1,
                'scale_pos_weight': 2,
                'max_depth': 4,
                'min_child_samples': 50,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'boosting_type': 'gbdt',
                'seed': self.random_seed + i,
            }
            params.update(self.lgb_params)
            params['seed'] = self.random_seed + i

            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, dtrain, num_boost_round=self.num_boost_round)
            self.models.append(model)

            if (i + 1) % 10 == 0:
                print(f"已完成 {i + 1}/{self.n_estimators} 个模型")

    def predict_proba(self, X):
        """对 hold-out 测试集做推断，不参与训练。"""
        if not self.models:
            raise ValueError("模型未训练，请先调用fit()方法")

        X = X[self.feature_names]

        all_preds = []
        for model in self.models:
            pred = model.predict(X)
            all_preds.append(pred)

        return np.mean(all_preds, axis=0)


def run_pu_learning_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = 'label',
    output_dir: str = 'data/results/pu_learning',
    n_estimators: int = 200,
    imbalance_ratio: float = 0.3,
    verbosity: int = -1,
    learning_rate: float = 0.05,
    num_leaves: int = 20,
    n_jobs: int = -1,
    scale_pos_weight: float = 2,
    max_depth: int = 4,
    min_child_samples: int = 50,
    subsample: float = 0.7,
    colsample_bytree: float = 0.7,
    num_boost_round: int = 1200,
    estimated_positive_rate: float = 0.1,
    save_prediction_files: bool = True,
    stop_event: Any = None,
) -> Dict:
    """
    运行 PU Learning 流水线。
    训练数据与测试数据严格分离：fit 只用 train_df，评估与预测只用 test_df。
    """
    os.makedirs(output_dir, exist_ok=True)
    train_df, test_df = _ensure_numeric_pu_frames(train_df, test_df, label_col)

    X_p = train_df[train_df[label_col] == 1].drop(columns=[label_col])
    y_p = train_df[train_df[label_col] == 1][label_col]

    X_u = train_df[train_df[label_col] == 0].drop(columns=[label_col])
    y_u = train_df[train_df[label_col] == 0][label_col]

    print("PU场景构建:")
    print(f"已知风险客户(P): {len(X_p)} 个")
    print(f"未标记数据(U): {len(X_u)} 个")

    lgb_params = {
        'verbosity': verbosity,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'n_jobs': n_jobs,
        'scale_pos_weight': scale_pos_weight,
        'max_depth': max_depth,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
    }
    pu_model = BaggingPULeaning(
        n_estimators=n_estimators,
        imbalance_ratio=imbalance_ratio,
        lgb_params=lgb_params,
        num_boost_round=num_boost_round,
    )
    pu_model.fit(X_p, X_u, y_p, y_u, stop_event=stop_event)

    _check_stop_event(stop_event)

    X_train = train_df.drop(columns=[label_col])
    train_probs = pu_model.predict_proba(X_train)
    train_df_result = train_df.copy()
    train_df_result['prob'] = train_probs
    if save_prediction_files:
        train_predictions_path = os.path.join(output_dir, 'train_predictions.csv')
        test_predictions_path = os.path.join(output_dir, 'test_predictions.csv')
    else:
        train_predictions_path = os.path.join(output_dir, AUTORESEARCH_TMP_TRAIN_PREDICTIONS)
        test_predictions_path = os.path.join(output_dir, AUTORESEARCH_TMP_TEST_PREDICTIONS)
    train_df_result.to_csv(train_predictions_path, index=False)
    if save_prediction_files:
        print(f"训练集预测已保存: {train_predictions_path}")
    else:
        print(f"训练集预测已写入临时文件: {train_predictions_path}")

    X_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col]

    test_probs = pu_model.predict_proba(X_test)

    threshold, predicted_positive_count = threshold_from_estimated_positive_rate(
        test_probs, estimated_positive_rate
    )
    metrics = compute_positive_metrics_at_threshold(
        y_test.values, test_probs, threshold
    )
    test_positive_count = int((y_test == 1).sum())
    test_evaluation = {
        'estimated_positive_rate': float(estimated_positive_rate),
        'predicted_positive_count': predicted_positive_count,
        'test_sample_count': len(test_probs),
        'test_positive_count': test_positive_count,
        **metrics,
    }
    print(f"预估正样本比例: {estimated_positive_rate:.2%}")
    print(f"测试集样本数: {len(test_probs)}, 判定为正例数: {predicted_positive_count}")
    print(f"推导阈值: {threshold:.4f}")
    print(f"测试集真实正样本数: {test_positive_count}")
    print(
        f"召回率: {metrics['recall']:.4f} | "
        f"精准率: {metrics['precision']:.4f} | F1: {metrics['f1']:.4f}"
    )

    test_df_result = test_df.copy()
    test_df_result['prob'] = test_probs
    test_df_result.to_csv(test_predictions_path, index=False)
    if save_prediction_files:
        print(f"测试集预测已保存: {test_predictions_path}")
    else:
        print(f"测试集预测已写入临时文件: {test_predictions_path}")

    return {
        'test_evaluation': test_evaluation,
        'predictions_path': test_predictions_path,
        'train_predictions_path': train_predictions_path,
    }


def run_pu_learning_pipeline_with_timeout(
    timeout_seconds: int = PU_RUN_TIMEOUT_SECONDS,
    stop_checker: Optional[Callable[[], bool]] = None,
    **pipeline_kwargs,
) -> Dict:
    """
    在独立子进程中运行 PU 流水线；超时或 stop_checker 返回 True 时终止子进程。
    数据加载在子进程内完成，以便停止/超时与 taskkill 行为一致。
    """
    global _active_pu_proc, _active_pu_stop_event, _active_pu_pid, _pu_user_stop_pending

    pipeline_kwargs.pop('stop_checker', None)
    output_dir = pipeline_kwargs.get('output_dir', 'data/results/pu_learning')
    estimated_positive_rate = float(pipeline_kwargs.get('estimated_positive_rate', 0.1))
    pu_train_params = extract_pu_train_params(pipeline_kwargs)
    os.makedirs(output_dir, exist_ok=True)

    if _user_stop_requested(stop_checker):
        _pu_user_stop_pending = False
        raise PUTrainStoppedError('用户已停止 PU Bagging 训练')

    ctx = mp.get_context('spawn')
    stop_event = ctx.Event()
    pipeline_kwargs['stop_event'] = stop_event
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_pu_pipeline_worker,
        args=(pipeline_kwargs, result_queue),
        daemon=True,
    )

    with _active_pu_proc_lock:
        _active_pu_proc = proc
        _active_pu_stop_event = stop_event
        _active_pu_pid = None

    proc.start()
    with _active_pu_proc_lock:
        _active_pu_pid = proc.pid

    watchdog_cancel, watchdog_thread = _start_pu_stop_watchdog(stop_checker, stop_event)

    if _user_stop_requested(stop_checker):
        watchdog_cancel.set()
        watchdog_thread.join(timeout=2)
        _kill_active_pu_process()
        with _active_pu_proc_lock:
            _active_pu_proc = None
            _active_pu_stop_event = None
            _active_pu_pid = None
        _pu_user_stop_pending = False
        raise PUTrainStoppedError('用户已停止 PU Bagging 训练')

    deadline = time.monotonic() + timeout_seconds
    stopped_by_user = False

    try:
        while proc.is_alive():
            if _user_stop_requested(stop_checker):
                stopped_by_user = True
                _kill_active_pu_process()
                break
            if time.monotonic() >= deadline:
                break
            proc.join(timeout=PU_STOP_POLL_INTERVAL_SECONDS)
    finally:
        watchdog_cancel.set()
        watchdog_thread.join(timeout=2)
        if stopped_by_user or _user_stop_requested(stop_checker):
            _kill_active_pu_process()
        with _active_pu_proc_lock:
            _active_pu_proc = None
            _active_pu_stop_event = None
            _active_pu_pid = None
        if stopped_by_user or _user_stop_requested(stop_checker):
            _pu_user_stop_pending = False

    if stopped_by_user or _user_stop_requested(stop_checker):
        raise PUTrainStoppedError('用户已停止 PU Bagging 训练')

    if proc.is_alive():
        stop_event.set()
        _terminate_process_tree(proc)
        _pu_user_stop_pending = False
        log_path = append_pu_train_log_timeout(
            output_dir, estimated_positive_rate, pu_train_params
        )
        raise PUTrainTimeoutError(
            f'PU Bagging 运行超过 {timeout_seconds // 60} 分钟已终止，记录已写入 {log_path}'
        )

    message = None
    while message is None:
        if _user_stop_requested(stop_checker):
            _kill_active_pu_process()
            raise PUTrainStoppedError('用户已停止 PU Bagging 训练')
        try:
            message = result_queue.get(timeout=PU_STOP_POLL_INTERVAL_SECONDS)
        except queue.Empty:
            if result_queue.empty():
                if _user_stop_requested(stop_checker):
                    raise PUTrainStoppedError('用户已停止 PU Bagging 训练')
                log_path = append_pu_train_log_timeout(
                    output_dir, estimated_positive_rate, pu_train_params
                )
                raise PUTrainTimeoutError(
                    f'PU Bagging 未返回结果（可能超时），记录已写入 {log_path}'
                )
            continue

    _pu_user_stop_pending = False
    if message['status'] == 'stopped':
        raise PUTrainStoppedError(message.get('error', '用户已停止 PU Bagging 训练'))
    if message['status'] == 'error':
        raise RuntimeError(message['error'])

    if _user_stop_requested(stop_checker):
        _pu_user_stop_pending = False
        raise PUTrainStoppedError('用户已停止 PU Bagging 训练')

    result = message['result']
    elapsed = message['elapsed']
    ev = result['test_evaluation']
    log_path = append_pu_train_log(
        output_dir,
        ev['estimated_positive_rate'],
        ev['threshold'],
        ev['recall'],
        ev['precision'],
        ev['f1'],
        f'{elapsed:.2f}s',
        params=pu_train_params,
    )
    result['pu_train_log_path'] = log_path
    result['run_time_seconds'] = elapsed
    return result
