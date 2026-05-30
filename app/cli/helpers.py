"""CLI 通用辅助函数。"""
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Optional

from app.services.data_core.pu.bagging import PU_RUN_TIMEOUT_SECONDS


def load_params_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f'参数 JSON 须为对象: {path}')
    return data


def parse_positive_rate(value: float) -> float:
    rate = float(value)
    if rate > 1:
        rate = rate / 100.0
    if not 0 < rate <= 1:
        raise ValueError('预估正样本比例须在 (0, 1] 或 (0, 100] 之间')
    return rate


def parse_timeout_seconds(
    timeout_seconds: Optional[int] = None,
    timeout_minutes: Optional[float] = None,
    *,
    default_seconds: int = PU_RUN_TIMEOUT_SECONDS,
) -> int:
    if timeout_seconds is not None:
        seconds = int(timeout_seconds)
    elif timeout_minutes is not None:
        seconds = int(float(timeout_minutes) * 60)
    else:
        seconds = default_seconds
    if seconds < 60:
        raise ValueError('超时时长至少为 60 秒（1 分钟）')
    if seconds > 86400:
        raise ValueError('超时时长不能超过 86400 秒（24 小时）')
    return seconds


def emit_result(data: Any, *, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    elif isinstance(data, dict):
        for key, value in data.items():
            print(f'{key}: {value}')
    else:
        print(data)


def fail(message: str, code: int = 1) -> None:
    print(f'错误: {message}', file=sys.stderr)
    raise SystemExit(code)


def uploads_dir(project_root: str) -> str:
    return os.path.join(project_root, 'data', 'uploads')


def results_dir(project_root: str) -> str:
    return os.path.join(project_root, 'data', 'results')


def pu_output_dir(project_root: str) -> str:
    return os.path.join(results_dir(project_root), 'pu_learning')


def fe_output_dir(project_root: str) -> str:
    return os.path.join(results_dir(project_root), 'feature_selection')


def load_train_csv(project_root: str, dataset_type: str, label_col: str):
    from app.services.data_core.shared.data_loader import DataLoader

    loader = DataLoader(label_col=label_col)
    upload = uploads_dir(project_root)
    if dataset_type == 'full':
        file_path = os.path.join(upload, 'full_dataset.csv')
        if not os.path.exists(file_path):
            file_path = os.path.join(project_root, 'data', 'train.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError('未找到 full_dataset.csv 或 data/train.csv')
        return loader._load_csv(file_path)
    train_path = os.path.join(upload, 'train_dataset.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError('未找到 train_dataset.csv，请先运行 dataset split')
    return loader._load_csv(train_path)


def create_flask_app():
    os.environ.setdefault('FINRISK_CLI', '1')
    from app import create_app

    return create_app()


def wait_autoresearch(
    get_status: Callable[[], Dict[str, Any]],
    stop_fn: Callable[[], Dict[str, Any]],
    *,
    prefix: str = 'autoresearch',
    poll_interval: float = 1.0,
) -> Dict[str, Any]:
    """前台轮询 autoresearch 日志，Ctrl+C 触发停止。"""
    seen = 0
    try:
        while True:
            status = get_status()
            logs = status.get('logs') or []
            for entry in logs[seen:]:
                msg = entry.get('message', '')
                print(f'[{prefix}] {msg}')
            seen = len(logs)
            if not status.get('running'):
                return status
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print(f'[{prefix}] 收到中断，正在发送停止请求…')
        stop_fn()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = get_status()
            logs = status.get('logs') or []
            for entry in logs[seen:]:
                print(f'[{prefix}] {entry.get("message", "")}')
            seen = len(logs)
            if not status.get('running'):
                return status
            time.sleep(poll_interval)
        return get_status()


def add_common_args(parser) -> None:
    parser.add_argument(
        '--project-root',
        default=os.getcwd(),
        help='项目根目录（默认当前目录）',
    )
    parser.add_argument(
        '--label-col',
        default=None,
        help='标签列名（默认读 dataset_preferences.json）',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='以 JSON 格式输出结果',
    )


def add_dataset_type_arg(parser, default: str = 'split') -> None:
    parser.add_argument(
        '--dataset',
        choices=('split', 'full'),
        default=default,
        help='数据集模式：split=train/test 分割；full=全量再划分',
    )


def add_rate_arg(parser, default: float = 10.0) -> None:
    parser.add_argument(
        '--rate',
        type=float,
        default=default,
        help='预估正样本比例，可填 10 表示 10%%',
    )


def add_timeout_args(parser) -> None:
    parser.add_argument(
        '--timeout-minutes',
        type=float,
        default=None,
        help='训练超时时长（分钟）',
    )
    parser.add_argument(
        '--timeout-seconds',
        type=int,
        default=None,
        help='训练超时时长（秒）',
    )
