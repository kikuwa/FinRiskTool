"""mlbase 子命令。"""
import os

from app.cli.context import CliContext
from app.cli.helpers import (
    add_common_args,
    add_dataset_type_arg,
    add_timeout_args,
    create_flask_app,
    emit_result,
    fail,
    fe_output_dir,
    load_params_json,
    parse_timeout_seconds,
    wait_autoresearch,
)
from app.services.data_core.feature_selection.pipeline import TOP_FEATURES_CSV
from app.services.data_core.mlbase.autoresearch import (
    get_mlbase_autoresearch_status,
    is_mlbase_autoresearch_running,
    request_mlbase_autoresearch_stop,
    start_mlbase_autoresearch,
)
from app.services.data_core.mlbase.comparison import run_mlbase_comparison
from app.services.data_core.mlbase.core import MLBASE_COMPARISON_JSON
from app.services.data_core.mlbase.param_optimizer import DEFAULT_MLBASE_PARAMS
from app.services.data_core.mlbase.test_eval import run_mlbase_test_evaluation
from app.services.data_core.mlbase.training_runner import (
    ML_RUN_TIMEOUT_SECONDS,
    MLBaseTrainTimeoutError,
    execute_mlbase_variant_training,
)
from app.services.data_core.shared.pu_session import resolve_dataset_type_from_pu_session


def _ml_params(args) -> dict:
    params = dict(DEFAULT_MLBASE_PARAMS)
    params.update(load_params_json(args.params_json))
    if args.recall_target is not None:
        params['recall_target'] = float(args.recall_target)
    return params


def _require_top_features(project_root: str) -> None:
    top_path = os.path.join(fe_output_dir(project_root), TOP_FEATURES_CSV)
    if not os.path.isfile(top_path):
        fail('未找到 top_features.csv，请先运行 fe confirm')


def cmd_compare(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    if is_mlbase_autoresearch_running():
        fail('MLBase autoresearch 正在运行，请先停止')
    _require_top_features(ctx.project_root)
    label_col = ctx.resolved_label_col()
    dataset_type = args.dataset or resolve_dataset_type_from_pu_session(ctx.project_root, 'split')
    ml_params = _ml_params(args)
    output_dir = fe_output_dir(ctx.project_root)

    comparison = run_mlbase_comparison(
        ctx.project_root,
        label_col=label_col,
        dataset_type=dataset_type,
        output_dir=output_dir,
        **ml_params,
    )
    emit_result({'success': True, 'comparison': comparison}, json_output=ctx.json_output)


def cmd_train(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    if args.variant == 'top_features':
        _require_top_features(ctx.project_root)
    label_col = ctx.resolved_label_col()
    dataset_type = args.dataset or resolve_dataset_type_from_pu_session(ctx.project_root, 'split')
    timeout = parse_timeout_seconds(
        args.timeout_seconds, args.timeout_minutes, default_seconds=ML_RUN_TIMEOUT_SECONDS,
    )
    ml_params = _ml_params(args)

    try:
        result = execute_mlbase_variant_training(
            ctx.project_root,
            variant=args.variant,
            label_col=label_col,
            dataset_type=dataset_type,
            ml_params=ml_params,
            timeout_seconds=timeout,
        )
    except MLBaseTrainTimeoutError as exc:
        fail(str(exc))

    emit_result(result, json_output=ctx.json_output)


def cmd_test(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    label_col = ctx.resolved_label_col()
    dataset_type = args.dataset or resolve_dataset_type_from_pu_session(ctx.project_root, 'split')
    comparison_path = os.path.join(fe_output_dir(ctx.project_root), MLBASE_COMPARISON_JSON)
    if not os.path.isfile(comparison_path):
        fail('未找到 mlbase_comparison.json，请先运行 mlbase compare')

    ml_params = _ml_params(args)
    metrics = run_mlbase_test_evaluation(
        ctx.project_root,
        variant=args.variant,
        label_col=label_col,
        dataset_type=dataset_type,
        reg_alpha=ml_params['reg_alpha'],
        reg_lambda=ml_params['reg_lambda'],
        learning_rate=ml_params['learning_rate'],
        n_estimators=int(ml_params['n_estimators']),
        max_depth=int(ml_params['max_depth']),
        min_child_samples=int(ml_params['min_child_samples']),
        subsample=ml_params['subsample'],
    )
    emit_result({'success': True, 'metrics': metrics}, json_output=ctx.json_output)


def cmd_autoresearch_start(args) -> None:
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY') or ''
    if args.variant == 'top_features':
        _require_top_features(args.project_root)

    ctx = CliContext(args.project_root, args.label_col, args.json)
    label_col = ctx.resolved_label_col()
    dataset_type = args.dataset or resolve_dataset_type_from_pu_session(ctx.project_root, 'split')
    timeout = parse_timeout_seconds(
        args.timeout_seconds, args.timeout_minutes, default_seconds=ML_RUN_TIMEOUT_SECONDS,
    )
    ml_params = _ml_params(args)

    app = create_flask_app()
    with app.app_context():
        project_root = app.config['PROJECT_ROOT']
        log_folder = app.config.get('LOG_FOLDER', os.path.join(project_root, 'logs'))
        result = start_mlbase_autoresearch(
            app,
            api_key=api_key,
            base_url=args.base_url or os.environ.get('OPENAI_BASE_URL'),
            model=args.model or os.environ.get('ML_PARAM_LLM_MODEL', 'gpt-4o-mini'),
            max_invalid_iterations=args.max_invalid,
            variant=args.variant,
            label_col=label_col,
            dataset_type=dataset_type,
            initial_params=ml_params,
            timeout_seconds=timeout,
            log_folder=log_folder,
        )
        if not result.get('success'):
            fail(result.get('error', '启动失败'))

        if args.wait:
            status = wait_autoresearch(
                get_mlbase_autoresearch_status,
                request_mlbase_autoresearch_stop,
                prefix='mlbase-autoresearch',
            )
            emit_result({'success': True, 'status': status}, json_output=ctx.json_output)
        else:
            emit_result({'success': True, 'message': 'autoresearch 已启动'}, json_output=ctx.json_output)


def cmd_autoresearch_stop(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    result = request_mlbase_autoresearch_stop()
    emit_result({**result, 'status': get_mlbase_autoresearch_status()}, json_output=ctx.json_output)


def cmd_autoresearch_status(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    emit_result(get_mlbase_autoresearch_status(), json_output=ctx.json_output)


def register_subcommands(subparsers) -> None:
    ml_parser = subparsers.add_parser('mlbase', help='MLBase 监督基线')
    ml_sub = ml_parser.add_subparsers(dest='ml_cmd', required=True)

    compare_parser = ml_sub.add_parser('compare', help='全量 vs Top 特征对比')
    add_common_args(compare_parser)
    add_dataset_type_arg(compare_parser)
    compare_parser.add_argument('--recall-target', type=float, default=None)
    compare_parser.add_argument('--params-json', default=None)
    compare_parser.set_defaults(handler=cmd_compare)

    train_parser = ml_sub.add_parser('train', help='单 variant 训练')
    add_common_args(train_parser)
    add_dataset_type_arg(train_parser)
    add_timeout_args(train_parser)
    train_parser.add_argument(
        '--variant',
        choices=('top_features', 'full_features'),
        default='top_features',
    )
    train_parser.add_argument('--recall-target', type=float, default=None)
    train_parser.add_argument('--params-json', default=None)
    train_parser.set_defaults(handler=cmd_train)

    test_parser = ml_sub.add_parser('test', help='Test 集评估')
    add_common_args(test_parser)
    add_dataset_type_arg(test_parser)
    test_parser.add_argument(
        '--variant',
        choices=('top_features', 'full_features'),
        default='top_features',
    )
    test_parser.add_argument('--recall-target', type=float, default=None)
    test_parser.add_argument('--params-json', default=None)
    test_parser.set_defaults(handler=cmd_test)

    ar_parser = ml_sub.add_parser('autoresearch', help='MLBase autoresearch 演进')
    ar_sub = ar_parser.add_subparsers(dest='ar_cmd', required=True)

    start_parser = ar_sub.add_parser('start', help='启动 autoresearch')
    add_common_args(start_parser)
    add_dataset_type_arg(start_parser)
    add_timeout_args(start_parser)
    start_parser.add_argument('--api-key', default=None)
    start_parser.add_argument('--base-url', default=None)
    start_parser.add_argument('--model', default=None)
    start_parser.add_argument('--max-invalid', type=int, default=3)
    start_parser.add_argument(
        '--variant',
        choices=('top_features', 'full_features'),
        default='top_features',
    )
    start_parser.add_argument('--recall-target', type=float, default=None)
    start_parser.add_argument('--params-json', default=None)
    start_parser.add_argument('--wait', action='store_true')
    start_parser.set_defaults(handler=cmd_autoresearch_start)

    stop_parser = ar_sub.add_parser('stop', help='停止 autoresearch')
    add_common_args(stop_parser)
    stop_parser.set_defaults(handler=cmd_autoresearch_stop)

    status_parser = ar_sub.add_parser('status', help='查询 autoresearch 状态')
    add_common_args(status_parser)
    status_parser.set_defaults(handler=cmd_autoresearch_status)
