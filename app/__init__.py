from flask import Flask, jsonify
import os
import traceback
import uuid
from datetime import datetime
from werkzeug.exceptions import HTTPException

from app.utils.session_logging import setup_session_logging
from app.utils.http_access_logging import configure_quiet_http_access_logs
from app.utils.workspace_reset import clear_data_workspace, should_clear_on_app_start


def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev')
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
    
    # Path Configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app.config['PROJECT_ROOT'] = project_root
    app.config['LOG_FOLDER'] = os.path.join(project_root, 'logs')
    app.config['DATA_FOLDER'] = os.path.join(project_root, 'data')
    app.config['RESULTS_FOLDER'] = os.path.join(project_root, 'results')
    # 每次进程启动唯一；前端据此在重启 run.py 后清空本地 LLM 凭证缓存
    app.config['SERVER_BOOT_ID'] = uuid.uuid4().hex

    if should_clear_on_app_start():
        cleared = clear_data_workspace(project_root)
        print(
            f'[workspace] cleared data/results and data/uploads '
            f'(boot={app.config["SERVER_BOOT_ID"]}, items={cleared})'
        )

    # Ensure directories exist
    os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)
    app.config['SESSION_LOG_FILE'] = setup_session_logging(app.config['LOG_FOLDER'])
    configure_quiet_http_access_logs()
    os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        
    # Register Blueprints
    from app.routes.main import main_bp
    from app.routes.data_tool import data_tool_bp
    from app.routes.risk_cot.generator import generator_bp
    from app.routes.risk_cot.inference import inference_bp
    from app.routes.risk_cot.inspector import inspector_bp
    from app.routes.risk_cot.views import views_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(data_tool_bp, url_prefix='/data_tool')
    app.register_blueprint(generator_bp)
    app.register_blueprint(inference_bp)
    app.register_blueprint(inspector_bp)
    app.register_blueprint(views_bp, url_prefix='/risk_cot')

    @app.context_processor
    def inject_server_boot_id():
        return {'server_boot_id': app.config['SERVER_BOOT_ID']}

    @app.route('/favicon.ico')
    def favicon():
        return '', 204

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        return jsonify({'error': e.name, 'message': e.description}), e.code

    # Register Global Error Handler
    @app.errorhandler(Exception)
    def handle_global_exception(e):
        error_time = datetime.now()
        error_type = type(e).__name__
        error_message = str(e)
        stack_trace = traceback.format_exc()
        
        # 1. 在控制台清晰打印
        print("="*80)
        print(f"CRITICAL ERROR at {error_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Type: {error_type}")
        print(f"Message: {error_message}")
        print(stack_trace)
        print("="*80)
        
        # 2. 写入会话日志（stdout/stderr 已 tee 到 SESSION_LOG_FILE）
        log_file_path = app.config.get('SESSION_LOG_FILE') or os.path.join(
            app.config['LOG_FOLDER'], f"error_{error_time.strftime('%Y%m%d')}.log"
        )
        log_content = (
            f"Timestamp: {error_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Error Type: {error_type}\n"
            f"Error Message: {error_message}\n"
            f"Stack Trace:\n{stack_trace}\n"
            f"{'-'*80}\n"
        )
        
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_content)
        except Exception as log_e:
            print(f"FATAL: Failed to write to log file: {log_e}")

        # 返回一个标准的 JSON 错误响应
        response = {
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred. The incident has been logged.'
        }
        return jsonify(response), 500
    
    return app
