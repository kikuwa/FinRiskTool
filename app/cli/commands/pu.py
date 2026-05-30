"""pu 子命令。"""
import os

from app.cli.context import CliContext
from app.cli.helpers import (
    add_common_args,
    add_dataset_type_arg,
    add_rate_arg,
    add_timeout_args,
    create_flask_app,
    emit_result,
    fail,
    load_params_json,
    parse_positive_rate,
    parse_timeout_seconds,
    wait_autoresearch,
)
from app.services.data_core.pu.autoresearch import (
    get_autoresearch_status,
    request_autoresearch_stop,
    start_autoresearch,
)
from app.services.data_core.pu.bagging import PUTrainTimeoutError
from app.services.data_core.pu.param_optimizer import DEFAULT_PU_PARAMS
from app.services.data_core.pu.training_runner import execute_pu_model_training
from app.services.data_core.shared.pu_session import (
    build_pu_session_payload,
    write_pu_session,
)


def _pu_params(args) -> dict:
    params = dict(DEFAULT_PU_PARAMS)
    params.update(load_params_json(args.params_json))
    return params


def cmd_train(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    label_col = ctx.resolved_label_col()
    rate = parse_positive_rate(args.rate)
    timeout = parse_timeout_seconds(args.timeout_seconds, args.timeout_minutes)
    pu_params = _pu_params(args)

    try:
        result = execute_pu_model_training(
            project_root=ctx.project_root,
            label_col=label_col,
            dataset_type=args.dataset,
            estimated_positive_rate=rate,
            pu_params=pu_params,
            num_boost_round=args.num_boost_round,
            timeout_seconds=timeout,
        )
    except PUTrainTimeoutError as exc:
        fail(str(exc))

    write_pu_session(ctx.project_root, build_pu_session_payload(
        ctx.project_root,
        estimated_positive_rate=rate,
        label_col=label_col,
        dataset_type=args.dataset,
    ))
    emit_result(result, json_output=ctx.json_output)


def cmd_autoresearch_start(args) -> None:
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        fail('请提供 --api-key 或设置 OPENAI_API_KEY')

    ctx = CliContext(args.project_root, args.label_col, args.json)
    label_col = ctx.resolved_label_col()
    rate = parse_positive_rate(args.rate)
    timeout = parse_timeout_seconds(args.timeout_seconds, args.timeout_minutes)
    pu_params = _pu_params(args)

    app = create_flask_app()
    with app.app_context():
        project_root = app.config['PROJECT_ROOT']
        log_folder = app.config.get('LOG_FOLDER', os.path.join(project_root, 'logs'))
        result = start_autoresearch(
            app,
            api_key=api_key,
            base_url=args.base_url or os.environ.get('OPENAI_BASE_URL'),
            model=args.model or os.environ.get('PU_PARAM_LLM_MODEL', 'gpt-4o-mini'),
            max_invalid_iterations=args.max_invalid,
            label_col=label_col,
            dataset_type=args.dataset,
            estimated_positive_rate=rate,
            initial_params=pu_params,
            num_boost_round=args.num_boost_round,
            timeout_seconds=timeout,
            log_folder=log_folder,
        )
        if not result.get('success'):
            fail(result.get('error', '启动失败'))

        write_pu_session(project_root, build_pu_session_payload(
            project_root,
            estimated_positive_rate=rate,
            label_col=label_col,
            dataset_type=args.dataset,
        ))

        if args.wait:
            status = wait_autoresearch(
                get_autoresearch_status,
                request_autoresearch_stop,
                prefix='pu-autoresearch',
            )
            emit_result({'success': True, 'status': status}, json_output=ctx.json_output)
        else:
            emit_result({'success': True, 'message': 'autoresearch 已启动'}, json_output=ctx.json_output)


def cmd_autoresearch_stop(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    result = request_autoresearch_stop()
    emit_result({**result, 'status': get_autoresearch_status()}, json_output=ctx.json_output)


def cmd_autoresearch_status(args) -> None:
    ctx = CliContext(args.project_root, args.label_col, args.json)
    emit_result(get_autoresearch_status(), json_output=ctx.json_output)


def register_subcommands(subparsers) -> None:
    pu_parser = subparsers.add_parser('pu', help='PU Bagging 训练与 autoresearch')
    pu_sub = pu_parser.add_subparsers(dest='pu_cmd', required=True)

    train_parser = pu_sub.add_parser('train', help='单次 PU 训练')
    add_common_args(train_parser)
    add_dataset_type_arg(train_parser)
    add_rate_arg(train_parser)
    add_timeout_args(train_parser)
    train_parser.add_argument('--params-json', default=None, help='PU 超参 JSON 文件')
    train_parser.add_argument('--num-boost-round', type=int, default=1200)
    train_parser.set_defaults(handler=cmd_train)

    ar_parser = pu_sub.add_parser('autoresearch', help='PU autoresearch 演进')
    ar_sub = ar_parser.add_subparsers(dest='ar_cmd', required=True)

    start_parser = ar_sub.add_parser('start', help='启动 autoresearch')
    add_common_args(start_parser)
    add_dataset_type_arg(start_parser)
    add_rate_arg(start_parser)
    add_timeout_args(start_parser)
    start_parser.add_argument('--api-key', default=None)
    start_parser.add_argument('--base-url', default=None)
    start_parser.add_argument('--model', default=None)
    start_parser.add_argument('--max-invalid', type=int, default=3)
    start_parser.add_argument('--params-json', default=None)
    start_parser.add_argument('--num-boost-round', type=int, default=1200)
    start_parser.add_argument('--wait', action='store_true', help='前台等待直至结束')
    start_parser.set_defaults(handler=cmd_autoresearch_start)

    stop_parser = ar_sub.add_parser('stop', help='停止 autoresearch')
    add_common_args(stop_parser)
    stop_parser.set_defaults(handler=cmd_autoresearch_stop)

    status_parser = ar_sub.add_parser('status', help='查询 autoresearch 状态')
    add_common_args(status_parser)
    status_parser.set_defaults(handler=cmd_autoresearch_status)
