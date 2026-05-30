"""开发服务器 HTTP 访问日志：正常 2xx/3xx 不写入会话日志。"""
import werkzeug.serving

_CONFIGURED = False
_ORIGINAL_LOG_REQUEST = None


def configure_quiet_http_access_logs(max_silent_status: int = 399) -> None:
    """
    抑制 Werkzeug 对成功响应的 access log（默认 <400 不打印）。
    4xx/5xx 仍会记录，便于排查问题。
    """
    global _CONFIGURED, _ORIGINAL_LOG_REQUEST
    if _CONFIGURED:
        return

    _ORIGINAL_LOG_REQUEST = werkzeug.serving.WSGIRequestHandler.log_request

    def log_request(self, code='-', size='-'):
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = None
        if status is not None and status <= max_silent_status:
            return
        _ORIGINAL_LOG_REQUEST(self, code, size)

    werkzeug.serving.WSGIRequestHandler.log_request = log_request
    _CONFIGURED = True
