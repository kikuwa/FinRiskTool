"""CLI 命令注册与入口。"""
import argparse
import os
import sys

from app.cli.commands import dataset, fe, mlbase, pu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='finrisk',
        description='finRiskTool 命令行工具：PU / 特征选择 / MLBase',
        epilog='完整文档: docs/CLI.md  参数参考: docs/cli/parameters.md',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    dataset.register_subcommands(subparsers)
    pu.register_subcommands(subparsers)
    fe.register_subcommands(subparsers)
    mlbase.register_subcommands(subparsers)
    return parser


def main(argv=None) -> int:
    os.environ.setdefault('FINRISK_CLI', '1')
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, 'handler', None)
    if handler is None:
        parser.print_help()
        return 1
    try:
        handler(args)
        return 0
    except SystemExit as exc:
        raise exc
    except Exception as exc:
        print(f'错误: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
