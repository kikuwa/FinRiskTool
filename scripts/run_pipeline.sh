#!/usr/bin/env bash
# finRiskTool 最小流水线示例（Linux / macOS）
# 用法: ./scripts/run_pipeline.sh [LABEL_COL]
# 示例: ./scripts/run_pipeline.sh RevolverStatus

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LABEL="${1:-RevolverStatus}"

python cli.py dataset split --label-col "$LABEL" --test-size 0.3
python cli.py pu train --dataset split --label-col "$LABEL" --rate 10 --timeout-minutes 20
python cli.py fe run --dataset split --label-col "$LABEL" --rate 10
python cli.py fe confirm --default
python cli.py mlbase compare --dataset split --recall-target 0.6
python cli.py mlbase test --variant top_features --dataset split

echo "流水线完成。详见 docs/CLI.md"
