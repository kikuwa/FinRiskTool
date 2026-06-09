"""fe（特征选择）子命令。"""
import os

import pandas as pd

from app.cli.context import CliContext
from app.cli.helpers import (
    add_common_args,
    add_dataset_type_arg,
    add_rate_arg,
    add_split_path_args,
    cli_split_paths_from_args,
    emit_result,
    fail,
    fe_output_dir,
    load_params_json,
    load_train_csv,
    parse_positive_rate,
    pu_output_dir,
)
from app.services.data_core.feature_selection.fe_optimizer import DEFAULT_FE_PARAMS
from app.services.data_core.feature_selection.pipeline import (
    FEATURE_CANDIDATES_CSV,
    confirm_top_features,
    run_feature_selection_pipeline,
)
from app.services.data_core.feature_selection.rfecv import run_rfecv_reselection
from app.services.data_core.shared.data_loader import DataLoader
from app.services.data_core.shared.pu_session import resolve_dataset_type_from_pu_session


def _fe_params(args) -> dict:
    params = dict(DEFAULT_FE_PARAMS)
    params.update(load_params_json(args.params_json))
    return params


def cmd_run(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    label_col = ctx.resolved_label_col()
    rate = parse_positive_rate(args.rate)
    dataset_type = args.dataset or resolve_dataset_type_from_pu_session(ctx.project_root, 'split')
    train_path, _ = cli_split_paths_from_args(args)
    pu_pred = os.path.join(pu_output_dir(ctx.project_root), 'train_predictions.csv')
    if not os.path.isfile(pu_pred):
        fail('未找到 train_predictions.csv，请先运行 pu train')

    loader = DataLoader(label_col=label_col)
    train_df = load_train_csv(
        ctx.project_root, dataset_type, label_col, train_path=train_path,
    )
    loader.validate_data(train_df)

    output_dir = fe_output_dir(ctx.project_root)
    result = run_feature_selection_pipeline(
        train_df=train_df,
        label_col=label_col,
        output_dir=output_dir,
        estimated_positive_rate=rate,
        train_predictions_path=pu_pred,
        project_root=ctx.project_root,
        fe_params=_fe_params(args),
    )
    emit_result({
        'success': True,
        'candidate_count': len(result.get('candidates', [])),
        'default_selected_count': len(result.get('default_selected', [])),
        'output_dir': output_dir,
        'feature_candidates_path': result.get('feature_candidates_path'),
    }, json_output=ctx.json_output)


def cmd_confirm(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    output_dir = fe_output_dir(ctx.project_root)
    cand_path = os.path.join(output_dir, FEATURE_CANDIDATES_CSV)
    if not os.path.isfile(cand_path):
        fail('未找到 feature_candidates.csv，请先运行 fe run')

    if args.default:
        df = pd.read_csv(cand_path)
        if 'selected' not in df.columns:
            fail('候选表无 selected 列')
        mask = df['selected'].apply(
            lambda v: str(v).strip().lower() in ('true', '1', 'yes') or v is True
        )
        features = df.loc[mask, 'feature_en'].astype(str).tolist()
    elif args.features:
        features = [f.strip() for f in args.features.split(',') if f.strip()]
    else:
        fail('请指定 --default 或 --features feat1,feat2,...')

    if not features:
        fail('未选中任何特征')

    path = confirm_top_features(output_dir, features)
    emit_result({
        'success': True,
        'path': path,
        'count': len(features),
    }, json_output=ctx.json_output)


def cmd_rfecv(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    label_col = ctx.resolved_label_col()
    dataset_type = args.dataset or resolve_dataset_type_from_pu_session(ctx.project_root, 'split')
    output_dir = fe_output_dir(ctx.project_root)

    def log_fn(msg, level='info'):
        print(f'[rfecv] {msg}')

    train_path, _ = cli_split_paths_from_args(args)
    result = run_rfecv_reselection(
        ctx.project_root,
        label_col=label_col,
        dataset_type=dataset_type,
        train_path=train_path,
        output_dir=output_dir,
        log_fn=log_fn,
    )
    emit_result({'success': True, **result}, json_output=ctx.json_output)


def register_subcommands(subparsers) -> None:
    fe_parser = subparsers.add_parser('fe', help='特征选择（PN + MI + LGB 稳定性）')
    fe_sub = fe_parser.add_subparsers(dest='fe_cmd', required=True)

    run_parser = fe_sub.add_parser('run', help='运行特征筛选流水线')
    add_common_args(run_parser)
    add_dataset_type_arg(run_parser)
    add_split_path_args(run_parser)
    add_rate_arg(run_parser)
    run_parser.add_argument('--params-json', default=None, help='特征工程超参 JSON')
    run_parser.set_defaults(handler=cmd_run)

    confirm_parser = fe_sub.add_parser('confirm', help='确认特征并写入 top_features.csv')
    add_common_args(confirm_parser)
    confirm_group = confirm_parser.add_mutually_exclusive_group(required=True)
    confirm_group.add_argument('--default', action='store_true', help='使用候选表默认选中项')
    confirm_group.add_argument('--features', default=None, help='逗号分隔特征名')
    confirm_parser.set_defaults(handler=cmd_confirm)

    rfecv_parser = fe_sub.add_parser('rfecv', help='RFECV 重选特征')
    add_common_args(rfecv_parser)
    add_dataset_type_arg(rfecv_parser)
    add_split_path_args(rfecv_parser)
    rfecv_parser.set_defaults(handler=cmd_rfecv)
