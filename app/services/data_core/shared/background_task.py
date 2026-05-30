"""Flask 后台线程任务通用 Runner（日志、状态、启停）。"""
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from flask import Flask

LogFn = Callable[[str, str], None]
EnrichStatusFn = Callable[[Dict[str, Any], Optional[str]], None]


class BackgroundTaskRunner:
    """封装 running/error/result/logs 与后台线程样板代码。"""

    def __init__(
        self,
        log_prefix: str,
        *,
        max_logs: int = 200,
        thread_name: str = 'background_task',
    ) -> None:
        self.log_prefix = log_prefix
        self.max_logs = max_logs
        self.thread_name = thread_name
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.error: Optional[str] = None
        self.result: Any = None
        self.logs: List[Dict[str, str]] = []

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def append_log(self, message: str, level: str = 'info') -> None:
        print(f'{self.log_prefix} {message}')
        with self._lock:
            entry = {
                'time': time.strftime('%H:%M:%S'),
                'message': message,
                'type': level,
            }
            self.logs.append(entry)
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs :]

    def _base_status_payload(self, log_since: int) -> Dict[str, Any]:
        logs = self.logs[log_since:] if log_since >= 0 else self.logs[-80:]
        return {
            'running': self.running,
            'error': self.error,
            'log_offset': len(self.logs),
            'logs': logs,
        }

    def get_status(
        self,
        log_since: int = 0,
        project_root: Optional[str] = None,
        enrich_fn: Optional[EnrichStatusFn] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            payload = self._base_status_payload(log_since)
            if enrich_fn:
                enrich_fn(payload, project_root)
            return payload

    def prepare_start(self, busy_message: str) -> Dict[str, Any]:
        with self._lock:
            if self.running:
                return {'success': False, 'error': busy_message}
            self.running = True
            self.error = None
            self.result = None
            self.logs.clear()
        return {'success': True, 'started': True}

    def start(
        self,
        app: Flask,
        worker: Callable[[], None],
        *,
        busy_message: str,
        pre_log: Optional[str] = None,
    ) -> Dict[str, Any]:
        prep = self.prepare_start(busy_message)
        if not prep.get('success'):
            return prep

        def wrapped() -> None:
            try:
                with app.app_context():
                    worker()
            except Exception as exc:
                msg = str(exc)
                self.append_log(f'失败: {msg}', 'error')
                print(traceback.format_exc())
                with self._lock:
                    self.error = msg
            finally:
                with self._lock:
                    self.running = False
                self._thread = None

        if pre_log:
            self.append_log(pre_log, 'info')
        self._thread = threading.Thread(target=wrapped, daemon=True, name=self.thread_name)
        self._thread.start()
        return prep

    def set_result(self, result: Any) -> None:
        with self._lock:
            self.result = result

    def set_error(self, msg: str) -> None:
        with self._lock:
            self.error = msg
