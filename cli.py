"""finRiskTool CLI 入口。"""
import os
import sys

# 确保项目根在 sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault('FINRISK_CLI', '1')

from app.cli.main import main

if __name__ == '__main__':
    raise SystemExit(main())
