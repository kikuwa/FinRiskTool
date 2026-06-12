"""dataset 子命令。"""
import os
import shutil

from app.cli.context import CliContext
from app.cli.helpers import emit_result, fail, uploads_dir
from app.services.data_core.shared.dataset_paths import resolve_full_dataset_path
from app.routes.data_tool import (
    save_dataset_analysis_report,
    save_dataset_preferences,
)
from app.services.data_core.dataset.data_analysis import analyze_dataset
from app.services.data_core.dataset.split_data import split_data
from app.services.data_core.shared.data_loader import DataLoader


def cmd_split(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    label_col = ctx.resolved_label_col()
    upload = uploads_dir(ctx.project_root)
    data_dir = os.path.join(ctx.project_root, 'data')

    try:
        input_path = resolve_full_dataset_path(ctx.project_root, full_path=args.input)
    except FileNotFoundError as exc:
        fail(str(exc))

    train_out = os.path.join(upload, 'train_dataset.csv')
    test_out = os.path.join(upload, 'test_dataset.csv')
    split_data(
        input_path,
        train_out,
        test_out,
        test_size=args.test_size,
        label_col=label_col,
    )
    save_dataset_preferences(
        ctx.project_root,
        label_col=label_col,
        test_size=args.test_size,
        full_dataset_path=input_path,
    )
    for src, dest in (
        (train_out, os.path.join(data_dir, 'train.csv')),
        (test_out, os.path.join(data_dir, 'test.csv')),
    ):
        try:
            shutil.copy2(src, dest)
        except PermissionError:
            pass

    emit_result({
        'success': True,
        'label_col': label_col,
        'train_path': train_out,
        'test_path': test_out,
        'test_size': args.test_size,
    }, json_output=ctx.json_output)


def cmd_analyze(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    label_col = ctx.resolved_label_col()
    try:
        input_path = resolve_full_dataset_path(ctx.project_root, full_path=args.input)
    except FileNotFoundError as exc:
        fail(str(exc))

    loader = DataLoader(label_col=label_col)
    df = loader._load_csv(input_path)
    save_dataset_preferences(
        ctx.project_root,
        label_col=label_col,
        full_dataset_path=input_path,
    )
    analysis = analyze_dataset(df, label_col=label_col)
    report_path = save_dataset_analysis_report(analysis, ctx.project_root)
    emit_result({
        'success': True,
        'label_col': label_col,
        'report_path': report_path,
        'rows': len(df),
    }, json_output=ctx.json_output)


def register_subcommands(subparsers) -> None:
    dataset_parser = subparsers.add_parser('dataset', help='数据集切分与分析')
    dataset_sub = dataset_parser.add_subparsers(dest='dataset_cmd', required=True)

    split_parser = dataset_sub.add_parser('split', help='分层切分 train/test')
    split_parser.add_argument('--input', default=None, help='全量 CSV 路径')
    split_parser.add_argument('--test-size', type=float, default=0.3, help='测试集比例')
    split_parser.add_argument('--project-root', default=os.getcwd())
    split_parser.add_argument('--label-col', default=None)
    split_parser.add_argument('--json', action='store_true')
    split_parser.set_defaults(handler=cmd_split)

    analyze_parser = dataset_sub.add_parser('analyze', help='生成 dataset_analysis_report.json')
    analyze_parser.add_argument('--input', default=None, help='全量 CSV 路径')
    analyze_parser.add_argument('--project-root', default=os.getcwd())
    analyze_parser.add_argument('--label-col', default=None)
    analyze_parser.add_argument('--json', action='store_true')
    analyze_parser.set_defaults(handler=cmd_analyze)
