"""CLI 运行上下文。"""
from dataclasses import dataclass
from typing import Optional

from app.services.data_core.shared.pu_session import resolve_label_col


@dataclass
class CliContext:
    project_root: str
    label_col: Optional[str] = None
    json_output: bool = False

    def resolved_label_col(self, fallback: str = 'label') -> str:
        if self.label_col and str(self.label_col).strip():
            return str(self.label_col).strip()
        return resolve_label_col(self.project_root, fallback)
