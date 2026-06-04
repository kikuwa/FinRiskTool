"""OpenAI 兼容客户端、JSON 解析与连接错误文案。"""
import json
import os
import re
import sys
from typing import Any, Dict, Optional

import httpx
import openai


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            raise ValueError('大模型返回内容无法解析为 JSON')
        return json.loads(match.group())


def ssl_verify_enabled() -> bool:
    """企业代理/MITM 场景可设置环境变量 OPENAI_SSL_VERIFY=false 跳过证书校验。"""
    val = os.environ.get('OPENAI_SSL_VERIFY', os.environ.get('PU_PARAM_SSL_VERIFY', 'true'))
    return val.strip().lower() not in ('0', 'false', 'no', 'off')


def normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    url = base_url.strip().rstrip('/')
    if url.endswith('/chat/completions'):
        url = url[: -len('/chat/completions')]
    if not url.endswith('/v1'):
        url = f'{url}/v1'
    return url


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    *,
    log_label: str = 'LLM',
) -> openai.OpenAI:
    verify_ssl = ssl_verify_enabled()
    timeout = float(os.environ.get('PU_PARAM_LLM_TIMEOUT', '180'))
    http_client = httpx.Client(verify=verify_ssl, timeout=timeout)
    client_kwargs: Dict[str, Any] = {'api_key': api_key or '', 'http_client': http_client}
    normalized = normalize_base_url(base_url)
    if normalized:
        client_kwargs['base_url'] = normalized
    print(
        f'[{log_label}] ssl_verify={verify_ssl}, timeout={timeout}s',
        file=sys.stderr,
    )
    return openai.OpenAI(**client_kwargs)


def format_llm_connection_error(exc: Exception) -> str:
    msg = str(exc)
    if 'CERTIFICATE_VERIFY_FAILED' in msg or 'self signed certificate' in msg.lower():
        return (
            'SSL 证书校验失败（常见于公司代理）。'
            '请在启动服务前设置环境变量 OPENAI_SSL_VERIFY=false 后重启，'
            '或配置企业根证书到 SSL_CERT_FILE。'
        )
    if 'ProxyError' in type(exc).__name__ or '504' in msg or 'Gateway Time-out' in msg:
        return (
            '经公司代理访问 API 超时。请检查 HTTP_PROXY/HTTPS_PROXY，'
            '或为 api.deepseek.com 配置 NO_PROXY，或更换可直连的网络。'
        )
    if 'ConnectError' in type(exc).__name__:
        return f'无法连接 API 服务：{msg}。请确认 API Base URL 为 https://api.deepseek.com/v1'
    return f'调用大模型 API 失败：{msg}'


def print_llm_response(
    content: str,
    log_folder: Optional[str] = None,
    *,
    label: str = 'LLM Response',
) -> None:
    banner = '=' * 80
    block = f'{banner}\n[{label}]\n{content}\n{banner}\n'
    print(block, file=sys.stderr)
