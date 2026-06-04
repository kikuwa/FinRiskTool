from flask import Blueprint, request, jsonify, current_app, send_from_directory, render_template, Response
import pandas as pd
import numpy as np
import os
import sys
import shutil
import traceback
import json
import chardet
import subprocess
from datetime import datetime
from app.services.data_core.shared.data_loader import DataLoader
from app.services.data_core.shared.pu_session import (
    build_pu_session_payload,
    read_pu_session,
    resolve_dataset_type_from_pu_session,
    resolve_label_col,
    resolve_label_col_from_pu_session,
    write_pu_session,
)
from app.services.data_core.dataset.split_data import split_data
from app.services.data_core.dataset.data_analysis import (
    analyze_dataset,
    drop_features_by_missing_ratio,
)
from app.services.data_core.pu.bagging import (
    PUTrainTimeoutError,
    PU_RUN_TIMEOUT_SECONDS,
    run_pu_learning_pipeline_with_timeout,
)
from app.services.data_core.pu.param_optimizer import (
    DEFAULT_PU_PARAMS,
    optimize_pu_params_with_llm,
)
from app.services.data_core.pu.training_runner import (
    execute_pu_model_training,
    load_pu_results_from_disk,
)
from app.services.data_core.pu.autoresearch import (
    get_autoresearch_status,
    load_pu_best_run,
    request_autoresearch_stop,
    start_autoresearch,
)
from app.services.data_core.feature_selection.pipeline import (
    FEATURE_CANDIDATES_CSV,
    TOP_FEATURES_CSV,
    confirm_top_features,
    read_top_features,
)
from app.services.data_core.feature_selection.tasks import (
    get_feature_selection_status,
    get_rfecv_status,
    start_feature_selection_task,
    start_rfecv_task,
)
from app.services.data_core.feature_selection.fe_optimizer import optimize_fe_params_with_llm
from app.services.data_core.mlbase.comparison import load_comparison_from_disk
from app.services.data_core.mlbase.tasks import (
    get_mlbase_comparison_status,
    get_mlbase_test_status,
    start_mlbase_comparison_task,
    start_mlbase_test_task,
)
from app.services.data_core.mlbase.test_eval import load_mlbase_test_metrics
from app.services.data_core.mlbase.param_optimizer import optimize_mlbase_params_with_llm
from app.services.data_core.mlbase.autoresearch import (
    get_mlbase_autoresearch_status,
    is_mlbase_autoresearch_running,
    request_mlbase_autoresearch_stop,
    start_mlbase_autoresearch,
)

DATASET_ANALYSIS_REPORT = 'dataset_analysis_report.json'

FEATURE_SELECTION_DOWNLOAD_WHITELIST = frozenset({
    FEATURE_CANDIDATES_CSV,
    TOP_FEATURES_CSV,
    'mlbase_comparison.json',
    'feature_selection_meta.json',
    'rfecv_meta.json',
    'mlbase_test_metrics.json',
    'mlbase_test_predictions.csv',
    'ml_train.tsv',
})
DATASET_PREFERENCES_FILE = 'dataset_preferences.json'


def _json_serialize_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def save_dataset_analysis_report(analysis_result, project_root: str) -> str:
    """将分析报告保存到 data/results/dataset_analysis_report.json"""
    results_dir = os.path.join(project_root, 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, DATASET_ANALYSIS_REPORT)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=4, ensure_ascii=False, default=_json_serialize_default)
    return report_path


def _preferences_path(project_root: str) -> str:
    return os.path.join(project_root, 'data', 'results', DATASET_PREFERENCES_FILE)


def load_dataset_preferences(project_root: str) -> dict:
    """读取已保存的标签列名、测试集比例等偏好。"""
    path = _preferences_path(project_root)
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    report_path = os.path.join(project_root, 'data', 'results', DATASET_ANALYSIS_REPORT)
    if os.path.isfile(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        label_col = report.get('label_col_requested')
        if label_col:
            return {'label_col': label_col, 'test_size': 0.3}

    return {'label_col': 'label', 'test_size': 0.3}


def save_dataset_preferences(
    project_root: str,
    label_col: str = None,
    test_size=None,
) -> dict:
    prefs = load_dataset_preferences(project_root)
    if label_col is not None and str(label_col).strip():
        prefs['label_col'] = str(label_col).strip()
    if test_size is not None:
        prefs['test_size'] = float(test_size)
    os.makedirs(os.path.dirname(_preferences_path(project_root)), exist_ok=True)
    with open(_preferences_path(project_root), 'w', encoding='utf-8') as f:
        json.dump(prefs, f, indent=4, ensure_ascii=False)
    return prefs


def _analyze_and_persist(df, label_col: str, project_root: str, missing_threshold_applied=None):
    label_col = (label_col or 'label').strip()
    save_dataset_preferences(project_root, label_col=label_col)
    analysis = analyze_dataset(
        df,
        label_col=label_col,
        missing_threshold_applied=missing_threshold_applied,
    )
    report_path = save_dataset_analysis_report(analysis, project_root)
    return analysis, report_path


def _full_dataset_path(project_root: str) -> str:
    return os.path.join(project_root, 'data', 'uploads', 'full_dataset.csv')


def _dataset_paths(project_root: str) -> dict:
    upload_dir = os.path.join(project_root, 'data', 'uploads')
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'data', 'results')
    return {
        'upload_dir': upload_dir,
        'data_dir': data_dir,
        'results_dir': results_dir,
        'full': os.path.join(upload_dir, 'full_dataset.csv'),
        'train_upload': os.path.join(upload_dir, 'train_dataset.csv'),
        'test_upload': os.path.join(upload_dir, 'test_dataset.csv'),
        'train': os.path.join(data_dir, 'train.csv'),
        'test': os.path.join(data_dir, 'test.csv'),
        'report': os.path.join(results_dir, DATASET_ANALYSIS_REPORT),
        'preferences': os.path.join(results_dir, DATASET_PREFERENCES_FILE),
    }


def _safe_unlink(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _detect_encoding(file_path: str) -> str:
    """
    使用 chardet 检测文件编码
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))
    return result['encoding']


def _csv_encoding_candidates(file_path: str) -> list:
    detected_encoding = _detect_encoding(file_path)
    encodings_to_try = []
    if detected_encoding:
        encodings_to_try.append(detected_encoding)
    for enc in ('utf-8', 'gbk', 'gb18030', 'big5', 'latin-1', 'utf-16', 'cp1252'):
        if enc not in encodings_to_try:
            encodings_to_try.append(enc)
    return encodings_to_try


def _read_csv_with_encodings(file_path: str, **read_csv_kwargs) -> pd.DataFrame:
    encodings_to_try = _csv_encoding_candidates(file_path)
    last_error = None
    for encoding in encodings_to_try:
        try:
            return pd.read_csv(file_path, encoding=encoding, low_memory=False, **read_csv_kwargs)
        except (UnicodeDecodeError, LookupError) as e:
            last_error = e
            continue
    try:
        return pd.read_csv(
            file_path, encoding='utf-8', errors='replace', low_memory=False, **read_csv_kwargs,
        )
    except Exception as e:
        raise ValueError(
            f"无法读取文件。尝试了以下编码: {encodings_to_try}。错误: {last_error or e}"
        ) from e


def _load_csv_preview(file_path: str, nrows: int = 5) -> pd.DataFrame:
    return _read_csv_with_encodings(file_path, nrows=nrows)


def _read_csv_column_names(file_path: str) -> list:
    return _read_csv_with_encodings(file_path, nrows=0).columns.tolist()


def _load_csv_robust(file_path: str) -> pd.DataFrame:
    """
    加载 CSV 文件，自动检测编码并包含多种编码回退机制
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    return _read_csv_with_encodings(file_path)

data_tool_bp = Blueprint('data_tool', __name__)

ALLOWED_EXTENSIONS = {'csv'}

def _sort_candidate_records(records: list) -> list:
    """按 rfecv_rank（若有）或 lgb_importance_mean 降序排序。"""

    def sort_key(row: dict):
        rfecv_score = row.get('rfecv_score')
        if rfecv_score not in (None, ''):
            try:
                return (0, -float(rfecv_score))
            except (TypeError, ValueError):
                pass
        rfecv_rank = row.get('rfecv_rank')
        if rfecv_rank not in (None, ''):
            try:
                return (1, int(float(rfecv_rank)))
            except (TypeError, ValueError):
                pass
        imp = row.get('lgb_importance_mean', 0)
        try:
            imp_val = -float(imp)
        except (TypeError, ValueError):
            imp_val = 0.0
        rank = row.get('rank', 999999)
        try:
            rank_val = int(float(rank)) if rank != '' else 999999
        except (TypeError, ValueError):
            rank_val = 999999
        return (2, imp_val, rank_val)

    records.sort(key=sort_key)
    return records


def _comparison_suggest_rfecv(comparison: dict) -> bool:
    summary = (comparison or {}).get('summary') or {}
    delta = summary.get('precision_delta')
    if delta is None:
        return False
    try:
        return float(delta) < 0
    except (TypeError, ValueError):
        return False


def log_exception(e):
    """一个简单的函数，用于在捕获异常时手动记录日志。"""
    error_time = datetime.now()
    error_type = type(e).__name__
    error_message = str(e)
    stack_trace = traceback.format_exc()
    
    # 打印到控制台
    print("="*80, file=sys.stderr)
    print(f"HANDLED EXCEPTION at {error_time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    print(f"Type: {error_type}", file=sys.stderr)
    print(f"Message: {error_message}", file=sys.stderr)
    print(stack_trace, file=sys.stderr)
    print("="*80, file=sys.stderr)
    
    # 写入会话日志（stderr 已通过 tee 镜像；此处再追加结构化记录）
    log_folder = current_app.config.get('LOG_FOLDER', os.path.join(current_app.config['PROJECT_ROOT'], 'logs'))
    log_file_path = current_app.config.get('SESSION_LOG_FILE') or os.path.join(
        log_folder, f"error_{error_time.strftime('%Y%m%d')}.log"
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
        print(f"FATAL: Failed to write to log file: {log_e}", file=sys.stderr)


def _append_session_log_line(level: str, line: str) -> None:
    """直接追加到会话日志，避免仅依赖 stdout/logger handler。"""
    log_file_path = current_app.config.get('SESSION_LOG_FILE')
    if not log_file_path:
        return
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    record = f'[{stamp}] [{level.upper()}] {line}\n'
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(record)
    except Exception as log_e:
        print(f"FATAL: Failed to append client log: {log_e}", file=sys.stderr)


@data_tool_bp.route('/client_log', methods=['POST'])
def client_log():
    """接收前端日志并同步到终端与会话日志。"""
    try:
        data = request.json or {}
        page = str(data.get('page', 'web')).strip()[:64]
        level = str(data.get('level', 'info')).strip().lower()[:16]
        message = str(data.get('message', '')).strip()
        if not message:
            return jsonify({'success': False, 'error': 'message 不能为空'}), 400

        # 防止日志被超长文本刷屏
        if len(message) > 2000:
            message = message[:2000] + '...<truncated>'

        tag = f'[client:{page}]'
        line = f'{tag} {message}'
        print(line)
        logger = current_app.logger
        if level == 'error':
            logger.error(line)
        elif level == 'warning':
            logger.warning(line)
        else:
            logger.info(line)
        _append_session_log_line(level, line)
        return jsonify({'success': True})
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Pages
@data_tool_bp.route('/')
def dataset_management():
    return render_template('data_tool/dataset.html', active_page='dataset')

@data_tool_bp.route('/pu_bagging')
def pu_bagging():
    return render_template('data_tool/pu_learning.html', active_page='data_engineering')

@data_tool_bp.route('/ensemble_feature_selection')
def ensemble_feature_selection():
    return render_template('data_tool/feature_engineering.html', active_page='feature_engineering')

# APIs
@data_tool_bp.route('/dataset_status', methods=['GET'])
def dataset_status():
    """返回服务端当前数据集文件与分析报告状态（供页面初始化同步，避免仅缓存前端状态）。"""
    try:
        project_root = current_app.config['PROJECT_ROOT']
        paths = _dataset_paths(project_root)
        prefs = load_dataset_preferences(project_root)
        label_col = prefs.get('label_col', 'label')
        test_size = prefs.get('test_size', 0.3)
        if not os.path.isfile(paths['preferences']):
            save_dataset_preferences(project_root, label_col=label_col, test_size=test_size)

        files = {
            'full': 'full_dataset.csv' if os.path.isfile(paths['full']) else '',
            'train': 'train_dataset.csv' if os.path.isfile(paths['train_upload']) else '',
            'test': 'test_dataset.csv' if os.path.isfile(paths['test_upload']) else '',
        }

        preview = []
        columns = []
        analysis = None
        report_path = None

        if os.path.isfile(paths['full']):
            preview_df = _load_csv_preview(paths['full'], nrows=5)
            preview = preview_df.fillna('').to_dict('records')
            columns = _read_csv_column_names(paths['full'])

            needs_analysis = True
            if os.path.isfile(paths['report']):
                with open(paths['report'], 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
                report_path = paths['report']
                if analysis.get('label_col_requested') == label_col:
                    needs_analysis = False

            if needs_analysis:
                df = _load_csv_robust(paths['full'])
                analysis, report_path = _analyze_and_persist(df, label_col, project_root)

        return jsonify({
            'success': True,
            'files': files,
            'preview': preview,
            'columns': columns,
            'analysis': analysis,
            'report_path': report_path,
            'label_col': label_col,
            'test_size': test_size,
        })
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/workflow_context', methods=['GET'])
def workflow_context():
    """三页 init 共用：标签列、数据集与各阶段磁盘结果摘要。"""
    try:
        project_root = current_app.config['PROJECT_ROOT']
        paths = _dataset_paths(project_root)
        prefs = load_dataset_preferences(project_root)
        label_col = resolve_label_col(project_root, prefs.get('label_col', 'label'))
        test_size = prefs.get('test_size', 0.3)

        files = {
            'full': 'full_dataset.csv' if os.path.isfile(paths['full']) else '',
            'train': 'train_dataset.csv' if os.path.isfile(paths['train_upload']) else '',
            'test': 'test_dataset.csv' if os.path.isfile(paths['test_upload']) else '',
        }

        pu_dir = os.path.join(project_root, 'data', 'results', 'pu_learning')
        fe_dir = os.path.join(project_root, 'data', 'results', 'feature_selection')
        train_pred = os.path.join(pu_dir, 'train_predictions.csv')
        test_pred = os.path.join(pu_dir, 'test_predictions.csv')
        session = read_pu_session(project_root)

        top_path = os.path.join(fe_dir, TOP_FEATURES_CSV)
        cand_path = os.path.join(fe_dir, FEATURE_CANDIDATES_CSV)
        confirmed = os.path.isfile(top_path)
        confirmed_count = 0
        if confirmed:
            try:
                confirmed_count = len(_load_csv_robust(top_path))
            except Exception:
                confirmed_count = 0

        ml_comp = os.path.join(fe_dir, 'mlbase_comparison.json')
        ml_test = os.path.join(fe_dir, 'mlbase_test_metrics.json')

        rate = None
        rate_pct = None
        if session:
            rate = session.get('estimated_positive_rate')
            if rate is not None and rate > 1:
                rate = rate / 100.0
            if rate is not None:
                rate_pct = rate * 100

        return jsonify({
            'success': True,
            'label_col': label_col,
            'test_size': test_size,
            'dataset_files': files,
            'has_analysis_report': os.path.isfile(paths['report']),
            'pu': {
                'has_session': session is not None,
                'has_train_predictions': os.path.isfile(train_pred),
                'has_test_predictions': os.path.isfile(test_pred),
                'estimated_positive_rate': rate,
                'estimated_positive_rate_pct': rate_pct,
                'dataset_type': (
                    session.get('dataset_type') if session else None
                ),
            },
            'feature_selection': {
                'has_candidates': os.path.isfile(cand_path),
                'confirmed': confirmed,
                'confirmed_count': confirmed_count,
            },
            'mlbase': {
                'has_comparison': os.path.isfile(ml_comp),
                'has_test_metrics': os.path.isfile(ml_test),
            },
        })
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/pu_last_result', methods=['GET'])
def pu_last_result():
    """从磁盘恢复上次 PU 训练结果摘要及最优参数。"""
    try:
        project_root = current_app.config['PROJECT_ROOT']
        result = load_pu_results_from_disk(project_root)
        best_run = load_pu_best_run(project_root)
        if not result and not best_run:
            return jsonify({
                'success': True,
                'has_result': False,
            })
        payload = {
            'success': True,
            'has_result': bool(result),
        }
        if result:
            payload['result'] = result
        if best_run:
            payload['best_f1'] = best_run.get('best_f1')
            payload['best_params'] = best_run.get('best_params')
        return jsonify(payload)
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/dataset_preferences', methods=['POST'])
def update_dataset_preferences():
    """保存标签列名、测试集比例（修改输入框时调用）。"""
    try:
        data = request.json or {}
        project_root = current_app.config['PROJECT_ROOT']
        prefs = save_dataset_preferences(
            project_root,
            label_col=data.get('label_col'),
            test_size=data.get('test_size'),
        )
        label_col = prefs['label_col']
        response = {
            'success': True,
            'label_col': label_col,
            'test_size': prefs['test_size'],
        }

        full_path = _full_dataset_path(project_root)
        if os.path.isfile(full_path):
            df = _load_csv_robust(full_path)
            analysis, report_path = _analyze_and_persist(df, label_col, project_root)
            response['analysis'] = analysis
            response['report_path'] = report_path
            response['preview'] = df.head(5).fillna('').to_dict('records')
            response['columns'] = df.columns.tolist()

        return jsonify(response)
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 400


@data_tool_bp.route('/clear_dataset', methods=['POST'])
def clear_dataset():
    """清空已上传数据集及分析报告（与页面「刷新状态」配合）。"""
    try:
        project_root = current_app.config['PROJECT_ROOT']
        paths = _dataset_paths(project_root)
        for key in ('full', 'train_upload', 'test_upload', 'train', 'test', 'report', 'preferences'):
            _safe_unlink(paths[key])
        return jsonify({'success': True, 'message': '数据集状态已清空'})
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/upload_full', methods=['POST'])
def upload_full():
    """上传完整数据集"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    label_col = request.form.get('label_col', 'label')
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        upload_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, 'full_dataset.csv')
        file.save(file_path)
        
        try:
            loader = DataLoader()
            df = loader._load_csv(file_path)
            preview_data = df.head(5).fillna('').to_dict('records')
            analysis_result, report_path = _analyze_and_persist(
                df, label_col, current_app.config['PROJECT_ROOT']
            )
            
            return jsonify({
                'success': '文件上传成功',
                'columns': df.columns.tolist(),
                'rows': len(df),
                'preview': preview_data,
                'analysis': analysis_result,
                'report_path': report_path,
            })
        except Exception as e:
            log_exception(e)
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': '只允许上传CSV文件'}), 400

@data_tool_bp.route('/upload_train', methods=['POST'])
def upload_train():
    """上传训练集"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    label_col = request.form.get('label_col', 'label')
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        upload_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads')
        data_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data')
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, 'train_dataset.csv')
        file.save(file_path)
        shutil.copy2(file_path, os.path.join(data_dir, 'train.csv'))
        
        try:
            loader = DataLoader()
            df = loader._load_csv(file_path)
            preview_data = df.head(5).fillna('').to_dict('records')
            analysis_result, report_path = _analyze_and_persist(
                df, label_col, current_app.config['PROJECT_ROOT']
            )
            
            return jsonify({
                'success': '文件上传成功',
                'columns': df.columns.tolist(),
                'rows': len(df),
                'preview': preview_data,
                'analysis': analysis_result,
                'report_path': report_path,
            })
        except Exception as e:
            log_exception(e)
            return jsonify({'error': str(e)}), 400
            
    return jsonify({'error': '只允许上传CSV文件'}), 400

@data_tool_bp.route('/upload_test', methods=['POST'])
def upload_test():
    """上传测试集"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    label_col = request.form.get('label_col', 'label')
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        upload_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads')
        data_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data')
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, 'test_dataset.csv')
        file.save(file_path)
        shutil.copy2(file_path, os.path.join(data_dir, 'test.csv'))
        
        try:
            loader = DataLoader()
            df = loader._load_csv(file_path)
            preview_data = df.head(5).fillna('').to_dict('records')
            analysis_result, report_path = _analyze_and_persist(
                df, label_col, current_app.config['PROJECT_ROOT']
            )
            
            return jsonify({
                'success': '文件上传成功',
                'columns': df.columns.tolist(),
                'rows': len(df),
                'preview': preview_data,
                'analysis': analysis_result,
                'report_path': report_path,
            })
        except Exception as e:
            log_exception(e)
            return jsonify({'error': str(e)}), 400
            
    return jsonify({'error': '只允许上传CSV文件'}), 400

@data_tool_bp.route('/apply_missing_threshold', methods=['POST'])
def apply_missing_threshold():
    """按缺失比例阈值删除特征，更新 full_dataset.csv 与分析报告"""
    try:
        data = request.json or {}
        label_col = data.get('label_col', 'label')
        threshold_pct = data.get('missing_threshold_pct', 70)

        try:
            threshold_pct = float(threshold_pct)
        except (TypeError, ValueError):
            return jsonify({'error': '无效的缺失比例阈值'}), 400

        if threshold_pct > 1:
            threshold_pct = threshold_pct / 100.0
        if not 0 <= threshold_pct <= 1:
            return jsonify({'error': '缺失比例阈值须在 0~100 之间'}), 400

        project_root = current_app.config['PROJECT_ROOT']
        full_path = _full_dataset_path(project_root)
        if not os.path.exists(full_path):
            return jsonify({'error': '全量数据集不存在，请先上传'}), 400

        df = _load_csv_robust(full_path)
        if label_col not in df.columns:
            return jsonify({'error': f"标签列 '{label_col}' 不存在"}), 400

        df_cleaned, dropped = drop_features_by_missing_ratio(
            df, max_missing_ratio=threshold_pct, label_col=label_col
        )
        if df_cleaned.shape[1] < 2:
            return jsonify({'error': '截断后特征过少，请降低阈值或检查数据'}), 400

        df_cleaned.to_csv(full_path, index=False)

        analysis_result, report_path = _analyze_and_persist(
            df_cleaned,
            label_col,
            project_root,
            missing_threshold_applied=threshold_pct,
        )
        preview_data = df_cleaned.head(5).fillna('').to_dict('records')

        return jsonify({
            'success': True,
            'message': f'已删除 {len(dropped)} 个缺失超过 {threshold_pct:.0%} 的特征',
            'dropped_features': dropped,
            'missing_threshold_pct': threshold_pct,
            'columns': df_cleaned.columns.tolist(),
            'rows': len(df_cleaned),
            'preview': preview_data,
            'analysis': analysis_result,
            'report_path': report_path,
        })
    except Exception as e:
        log_exception(e)
        return jsonify({'error': str(e)}), 500

@data_tool_bp.route('/split_dataset', methods=['POST'])
def split_dataset():
    """分割全量数据集"""
    try:
        data = request.json or {}
        label_col = data.get('label_col', 'label')
        test_size = data.get('test_size', 0.3)
        
        upload_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads')
        data_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data')
        
        full_path = os.path.join(upload_dir, 'full_dataset.csv')
        
        if not os.path.exists(full_path):
            return jsonify({'error': '全量数据集不存在，请先上传'}), 400

        project_root = current_app.config['PROJECT_ROOT']
        save_dataset_preferences(project_root, label_col=label_col, test_size=test_size)
            
        train_upload = os.path.join(upload_dir, 'train_dataset.csv')
        test_upload = os.path.join(upload_dir, 'test_dataset.csv')

        # 优先写入 uploads（PU Learning 等流程主要使用此路径）
        split_data(full_path, train_upload, test_upload, label_col=label_col, test_size=test_size)

        data_sync_warnings = []
        for src, dest in (
            (train_upload, os.path.join(data_dir, 'train.csv')),
            (test_upload, os.path.join(data_dir, 'test.csv')),
        ):
            try:
                shutil.copy2(src, dest)
            except PermissionError:
                data_sync_warnings.append(os.path.basename(dest))
        
        df = _load_csv_robust(full_path)
        analysis_result, report_path = _analyze_and_persist(
            df, label_col, current_app.config['PROJECT_ROOT']
        )
        
        resp = {
            'success': '数据集分割成功',
            'analysis': analysis_result,
            'report_path': report_path,
        }
        if data_sync_warnings:
            locked = '、'.join(f'data/{name}' for name in data_sync_warnings)
            resp['warning'] = (
                f"训练/测试集已写入 data/uploads，但无法同步到 {locked}（文件可能被占用）。"
                f"请关闭 Excel 后重新分割；PU Learning 可直接使用 uploads 下的文件。"
            )
        return jsonify(resp)

    except PermissionError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        log_exception(e)
        return jsonify({'error': f'分割失败: {str(e)}'}), 500

@data_tool_bp.route('/download_analysis_report', methods=['POST'])
def download_analysis_report():
    """下载分析报告"""
    try:
        data = request.json or {}
        analysis_result = data.get('analysis_result')
        
        if not analysis_result:
            return jsonify({'error': '没有分析结果'}), 400

        save_dataset_analysis_report(analysis_result, current_app.config['PROJECT_ROOT'])
        report_content = json.dumps(
            analysis_result, indent=4, ensure_ascii=False, default=_json_serialize_default
        )
        
        return Response(
            report_content,
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment;filename=dataset_analysis_report.json'}
        )
    except Exception as e:
        log_exception(e)
        return jsonify({'error': str(e)}), 400

def _parse_estimated_positive_rate(data: dict):
    estimated_positive_rate = data.get('estimated_positive_rate')
    if estimated_positive_rate is None:
        raise ValueError('请填写预估正样本比例')
    try:
        estimated_positive_rate = float(estimated_positive_rate)
    except (TypeError, ValueError):
        raise ValueError('预估正样本比例格式无效')
    if estimated_positive_rate > 1:
        estimated_positive_rate = estimated_positive_rate / 100.0
    if not 0 < estimated_positive_rate <= 1:
        raise ValueError('预估正样本比例须在 0~100% 之间')
    return estimated_positive_rate


def _parse_timeout_seconds(data: dict) -> int:
    """解析训练超时时长（秒），支持 timeout_seconds 或 timeout_minutes。"""
    if data.get('timeout_seconds') is not None:
        try:
            seconds = int(float(data['timeout_seconds']))
        except (TypeError, ValueError):
            raise ValueError('超时时长（秒）格式无效')
    elif data.get('timeout_minutes') is not None:
        try:
            seconds = int(float(data['timeout_minutes']) * 60)
        except (TypeError, ValueError):
            raise ValueError('超时时长（分钟）格式无效')
    else:
        seconds = PU_RUN_TIMEOUT_SECONDS
    if seconds < 60:
        raise ValueError('超时时长至少为 60 秒（1 分钟）')
    if seconds > 86400:
        raise ValueError('超时时长不能超过 86400 秒（24 小时）')
    return seconds


def _pu_params_from_request(data: dict) -> dict:
    return {
        'n_estimators': data.get('n_estimators', 200),
        'imbalance_ratio': data.get('imbalance_ratio', 0.2),
        'verbosity': data.get('verbosity', -1),
        'learning_rate': data.get('learning_rate', 0.05),
        'num_leaves': data.get('num_leaves', 20),
        'n_jobs': data.get('n_jobs', -1),
        'scale_pos_weight': data.get('scale_pos_weight', 2),
        'max_depth': data.get('max_depth', 4),
        'min_child_samples': data.get('min_child_samples', 50),
        'subsample': data.get('subsample', 0.7),
        'colsample_bytree': data.get('colsample_bytree', 0.7),
    }


@data_tool_bp.route('/run_model', methods=['POST'])
def run_model():
    """运行 PU Learning 模型"""
    try:
        data = request.json or {}
        project_root = current_app.config['PROJECT_ROOT']
        label_col = resolve_label_col(project_root, data.get('label_col', 'label'))
        dataset_type = data.get('dataset_type', 'full')
        estimated_positive_rate = _parse_estimated_positive_rate(data)
        pu_params = _pu_params_from_request(data)
        timeout_seconds = _parse_timeout_seconds(data)

        result = execute_pu_model_training(
            project_root=project_root,
            label_col=label_col,
            dataset_type=dataset_type,
            estimated_positive_rate=estimated_positive_rate,
            pu_params=pu_params,
            num_boost_round=data.get('num_boost_round', 1200),
            timeout_seconds=timeout_seconds,
        )
        write_pu_session(project_root, build_pu_session_payload(
            project_root,
            estimated_positive_rate=estimated_positive_rate,
            label_col=label_col,
            dataset_type=dataset_type,
        ))
        return jsonify(result)

    except PUTrainTimeoutError as e:
        log_exception(e)
        return jsonify({
            'success': False,
            'error': str(e),
            'timed_out': True,
        }), 408

    except ValueError as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 400

    except Exception as e:
        log_exception(e)
        return jsonify({
            'success': False,
            'error': str(e)
        })


@data_tool_bp.route('/pu_autoresearch/start', methods=['POST'])
def pu_autoresearch_start():
    """启动 autoresearch 演进循环。"""
    try:
        data = request.json or {}
        api_key = data.get('api_key') or ''

        max_invalid = data.get('max_invalid_iterations', 3)
        try:
            max_invalid = int(max_invalid)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': '最大无效迭代次数格式无效'}), 400
        if max_invalid < 1:
            return jsonify({'success': False, 'error': '最大无效迭代次数至少为 1'}), 400

        estimated_positive_rate = _parse_estimated_positive_rate(data)
        pu_params = _pu_params_from_request(data)
        timeout_seconds = _parse_timeout_seconds(data)
        base_url = data.get('base_url') or os.environ.get('OPENAI_BASE_URL')
        model = data.get('model') or os.environ.get('PU_PARAM_LLM_MODEL', 'gpt-4o-mini')
        project_root = current_app.config['PROJECT_ROOT']
        log_folder = current_app.config.get(
            'LOG_FOLDER', os.path.join(project_root, 'logs')
        )

        label_col = resolve_label_col(project_root, data.get('label_col', 'label'))
        dataset_type = data.get('dataset_type', 'split')
        result = start_autoresearch(
            current_app._get_current_object(),
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_invalid_iterations=max_invalid,
            label_col=label_col,
            dataset_type=dataset_type,
            estimated_positive_rate=estimated_positive_rate,
            initial_params=pu_params or DEFAULT_PU_PARAMS,
            num_boost_round=data.get('num_boost_round', 1200),
            timeout_seconds=timeout_seconds,
            log_folder=log_folder,
        )
        if not result.get('success'):
            return jsonify(result), 409
        write_pu_session(project_root, build_pu_session_payload(
            project_root,
            estimated_positive_rate=estimated_positive_rate,
            label_col=label_col,
            dataset_type=dataset_type,
        ))
        return jsonify({**result, 'status': get_autoresearch_status()})
    except ValueError as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/pu_autoresearch/stop', methods=['POST'])
def pu_autoresearch_stop():
    """请求停止 autoresearch。"""
    try:
        result = request_autoresearch_stop()
        return jsonify({**result, 'status': get_autoresearch_status()})
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/pu_autoresearch/status', methods=['GET'])
def pu_autoresearch_status():
    """查询 autoresearch 运行状态。"""
    since = request.args.get('since', 0, type=int)
    status = get_autoresearch_status()
    logs = status.get('logs', [])
    if since > 0:
        status['logs'] = logs[since:]
    else:
        status['logs'] = logs[-80:]
    status['log_offset'] = len(logs)
    return jsonify({'success': True, **status})


@data_tool_bp.route('/pu_session_config', methods=['GET'])
def pu_session_config():
    """读取 PU 会话配置（预估正样本比例等）。"""
    project_root = current_app.config['PROJECT_ROOT']
    label_col = resolve_label_col(project_root)
    session = read_pu_session(project_root)
    if not session:
        return jsonify({
            'success': True,
            'has_session': False,
            'label_col': label_col,
        })
    rate = session.get('estimated_positive_rate')
    if rate is not None and rate > 1:
        rate = rate / 100.0
    dataset_type = session.get('dataset_type', 'split')
    return jsonify({
        'success': True,
        'has_session': True,
        'estimated_positive_rate': rate,
        'estimated_positive_rate_pct': (rate * 100) if rate is not None else None,
        'label_col': label_col,
        'dataset_type': dataset_type,
    })


@data_tool_bp.route('/run_model_feature_selection', methods=['POST'])
def run_model_feature_selection():
    """提交特征选择后台任务（立即返回，避免浏览器长时间等待断开）。"""
    try:
        data = request.json or {}
        project_root = current_app.config['PROJECT_ROOT']
        label_col = resolve_label_col_from_pu_session(project_root)
        dataset_type = resolve_dataset_type_from_pu_session(
            project_root, data.get('dataset_type', 'split')
        )
        estimated_positive_rate = _parse_estimated_positive_rate(data)
        fe_params = data.get('fe_params')
        if fe_params is not None and not isinstance(fe_params, dict):
            return jsonify({'success': False, 'error': 'fe_params 须为对象'}), 400

        pu_pred_path = os.path.join(
            project_root, 'data', 'results', 'pu_learning', 'train_predictions.csv'
        )
        if not os.path.exists(pu_pred_path):
            return jsonify({
                'success': False,
                'error': '未找到 train_predictions.csv，请先在 PU Learning 完成训练',
            }), 400

        result = start_feature_selection_task(
            current_app._get_current_object(),
            project_root=project_root,
            label_col=label_col,
            dataset_type=dataset_type,
            estimated_positive_rate=estimated_positive_rate,
            fe_params=fe_params,
        )
        if not result.get('success'):
            return jsonify(result), 409
        return jsonify({
            **result,
            'label_col': label_col,
            'dataset_type': dataset_type,
        })

    except ValueError as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)})


@data_tool_bp.route('/feature_selection/status', methods=['GET'])
def feature_selection_status():
    """轮询特征选择后台任务状态。"""
    since = request.args.get('log_since', 0, type=int)
    project_root = current_app.config['PROJECT_ROOT']
    status = get_feature_selection_status(log_since=since, project_root=project_root)
    return jsonify({'success': True, **status})


@data_tool_bp.route('/feature_selection/candidates', methods=['GET'])
def feature_selection_candidates():
    """获取 MI 后候选特征列表。"""
    output_dir = os.path.join(
        current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection'
    )
    path = os.path.join(output_dir, FEATURE_CANDIDATES_CSV)
    if not os.path.exists(path):
        return jsonify({'success': False, 'error': '请先运行特征筛选'}), 404
    df = _load_csv_robust(path)
    df = df.fillna('')
    records = df.to_dict(orient='records')
    for row in records:
        if 'rank' in row and row['rank'] != '':
            try:
                row['rank'] = int(float(row['rank']))
            except (TypeError, ValueError):
                row['rank'] = ''
        if 'selected' in row:
            val = row['selected']
            if isinstance(val, str):
                row['selected'] = val.strip().lower() in ('true', '1', 'yes')
            else:
                row['selected'] = bool(val) if pd.notna(val) else False
        if 'rfecv_score' in row and row['rfecv_score'] != '':
            try:
                row['rfecv_score'] = float(row['rfecv_score'])
            except (TypeError, ValueError):
                pass
        if 'rfecv_rank' in row and row['rfecv_rank'] != '':
            try:
                row['rfecv_rank'] = int(float(row['rfecv_rank']))
            except (TypeError, ValueError):
                row['rfecv_rank'] = ''
    meta_path = os.path.join(output_dir, 'feature_selection_meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    records = _sort_candidate_records(records)
    return jsonify({'success': True, 'candidates': records, 'meta': meta})


@data_tool_bp.route('/feature_selection/confirmed', methods=['GET'])
def feature_selection_confirmed():
    """是否已确认 top_features.csv。"""
    output_dir = os.path.join(
        current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection'
    )
    top_path = os.path.join(output_dir, TOP_FEATURES_CSV)
    if not os.path.isfile(top_path):
        return jsonify({
            'success': True,
            'confirmed': False,
            'count': 0,
            'features': [],
        })
    features = read_top_features(output_dir)
    return jsonify({
        'success': True,
        'confirmed': True,
        'count': len(features),
        'features': features,
        'available_downloads': [
            f for f in FEATURE_SELECTION_DOWNLOAD_WHITELIST
            if os.path.isfile(os.path.join(output_dir, f))
        ],
    })


@data_tool_bp.route('/feature_selection/confirm', methods=['POST'])
def feature_selection_confirm():
    """业务确认最终特征列表。"""
    try:
        data = request.json or {}
        selected = data.get('selected_features') or []
        if not selected:
            return jsonify({'success': False, 'error': '请至少选择一个特征'}), 400
        output_dir = os.path.join(
            current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection'
        )
        path = confirm_top_features(output_dir, selected)
        return jsonify({
            'success': True,
            'path': path,
            'count': len(selected),
        })
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 400


def _ml_params_from_request(data: dict) -> dict:
    return {
        'recall_target': float(data.get('recall_target', 0.5)),
        'reg_alpha': float(data.get('reg_alpha', 0.1)),
        'reg_lambda': float(data.get('reg_lambda', 0.1)),
        'learning_rate': float(data.get('learning_rate', 0.05)),
        'n_estimators': int(data.get('n_estimators', 150)),
        'max_depth': int(data.get('max_depth', 6)),
        'min_child_samples': int(data.get('min_child_samples', 50)),
        'subsample': float(data.get('subsample', 0.8)),
    }


@data_tool_bp.route('/mlbase_autoresearch/start', methods=['POST'])
def mlbase_autoresearch_start():
    """启动 MLBase autoresearch 演进循环。"""
    try:
        data = request.json or {}
        api_key = data.get('api_key') or ''

        max_invalid = data.get('max_invalid_iterations', 3)
        try:
            max_invalid = int(max_invalid)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': '最大无效迭代次数格式无效'}), 400
        if max_invalid < 1:
            return jsonify({'success': False, 'error': '最大无效迭代次数至少为 1'}), 400

        variant = data.get('variant', 'top_features')
        if variant not in ('full_features', 'top_features'):
            return jsonify({'success': False, 'error': 'variant 须为 full_features 或 top_features'}), 400

        project_root = current_app.config['PROJECT_ROOT']
        output_dir = os.path.join(project_root, 'data', 'results', 'feature_selection')
        top_path = os.path.join(output_dir, TOP_FEATURES_CSV)
        if variant == 'top_features' and not os.path.isfile(top_path):
            return jsonify({
                'success': False,
                'error': '请先确认特征（top_features.csv 不存在）',
            }), 400

        timeout_seconds = _parse_timeout_seconds(data)
        base_url = data.get('base_url') or os.environ.get('OPENAI_BASE_URL')
        model = data.get('model') or os.environ.get('ML_PARAM_LLM_MODEL', 'gpt-4o-mini')
        log_folder = current_app.config.get(
            'LOG_FOLDER', os.path.join(project_root, 'logs')
        )
        label_col = resolve_label_col(project_root, data.get('label_col', 'label'))
        dataset_type = resolve_dataset_type_from_pu_session(
            project_root, data.get('dataset_type', 'split')
        )

        result = start_mlbase_autoresearch(
            current_app._get_current_object(),
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_invalid_iterations=max_invalid,
            variant=variant,
            label_col=label_col,
            dataset_type=dataset_type,
            initial_params=_ml_params_from_request(data),
            timeout_seconds=timeout_seconds,
            log_folder=log_folder,
        )
        if not result.get('success'):
            return jsonify(result), 409
        return jsonify({**result, 'status': get_mlbase_autoresearch_status()})
    except ValueError as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/mlbase_autoresearch/stop', methods=['POST'])
def mlbase_autoresearch_stop():
    """请求停止 MLBase autoresearch。"""
    try:
        result = request_mlbase_autoresearch_stop()
        return jsonify({**result, 'status': get_mlbase_autoresearch_status()})
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/mlbase_autoresearch/status', methods=['GET'])
def mlbase_autoresearch_status():
    """查询 MLBase autoresearch 运行状态。"""
    since = request.args.get('since', 0, type=int)
    status = get_mlbase_autoresearch_status()
    logs = status.get('logs', [])
    if since > 0:
        status['logs'] = logs[since:]
    else:
        status['logs'] = logs[-80:]
    status['log_offset'] = len(logs)
    return jsonify({'success': True, **status})


@data_tool_bp.route('/run_mlbase_comparison', methods=['POST'])
def run_mlbase_comparison():
    """提交全量 vs Top 特征 MLBase 对比后台任务。"""
    try:
        if is_mlbase_autoresearch_running():
            return jsonify({
                'success': False,
                'error': 'MLBase autoresearch 正在运行，请先停止',
            }), 409
        data = request.json or {}
        project_root = current_app.config['PROJECT_ROOT']
        label_col = resolve_label_col_from_pu_session(project_root)
        dataset_type = resolve_dataset_type_from_pu_session(
            project_root, data.get('dataset_type', 'split')
        )
        output_dir = os.path.join(project_root, 'data', 'results', 'feature_selection')
        top_path = os.path.join(output_dir, TOP_FEATURES_CSV)
        if not os.path.exists(top_path):
            return jsonify({
                'success': False,
                'error': '请先确认特征（top_features.csv 不存在）',
            }), 400

        run_kwargs = dict(
            label_col=label_col,
            dataset_type=dataset_type,
            recall_target=float(data.get('recall_target', 0.5)),
            reg_alpha=float(data.get('reg_alpha', 0.1)),
            reg_lambda=float(data.get('reg_lambda', 0.1)),
            learning_rate=float(data.get('learning_rate', 0.05)),
            n_estimators=int(data.get('n_estimators', 150)),
            max_depth=int(data.get('max_depth', 6)),
            min_child_samples=int(data.get('min_child_samples', 50)),
            subsample=float(data.get('subsample', 0.8)),
            output_dir=output_dir,
        )
        result = start_mlbase_comparison_task(
            current_app._get_current_object(),
            project_root=project_root,
            run_kwargs=run_kwargs,
        )
        if not result.get('success'):
            return jsonify(result), 409
        return jsonify(result)
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/mlbase_comparison/status', methods=['GET'])
def mlbase_comparison_status():
    """轮询 ML 对比后台任务。"""
    since = request.args.get('log_since', 0, type=int)
    project_root = current_app.config['PROJECT_ROOT']
    status = get_mlbase_comparison_status(log_since=since, project_root=project_root)
    comparison = status.get('comparison')
    if comparison is None and project_root:
        comparison = load_comparison_from_disk(project_root)
    status['suggest_rfecv'] = _comparison_suggest_rfecv(comparison or {})
    return jsonify({'success': True, **status})


@data_tool_bp.route('/mlbase_comparison/result', methods=['GET'])
def mlbase_comparison_result():
    """读取已保存的 mlbase_comparison.json。"""
    project_root = current_app.config['PROJECT_ROOT']
    comparison = load_comparison_from_disk(project_root)
    if not comparison:
        return jsonify({'success': False, 'error': '尚无 ML 对比结果'}), 404
    return jsonify({'success': True, 'comparison': comparison})


@data_tool_bp.route('/run_mlbase_test_eval', methods=['POST'])
def run_mlbase_test_eval():
    """使用 ML 对比确定的模型方案与阈值，在 test 集上评估。"""
    try:
        data = request.json or {}
        project_root = current_app.config['PROJECT_ROOT']
        label_col = resolve_label_col_from_pu_session(project_root)
        dataset_type = resolve_dataset_type_from_pu_session(
            project_root, data.get('dataset_type', 'split')
        )
        variant = data.get('variant', 'top_features')
        if variant not in ('full_features', 'top_features'):
            return jsonify({'success': False, 'error': 'variant 须为 full_features 或 top_features'}), 400

        output_dir = os.path.join(project_root, 'data', 'results', 'feature_selection')
        run_kwargs = dict(
            variant=variant,
            label_col=label_col,
            dataset_type=dataset_type,
            output_dir=output_dir,
            reg_alpha=float(data.get('reg_alpha', 0.1)),
            reg_lambda=float(data.get('reg_lambda', 0.1)),
            learning_rate=float(data.get('learning_rate', 0.05)),
            n_estimators=int(data.get('n_estimators', 150)),
            max_depth=int(data.get('max_depth', 6)),
            min_child_samples=int(data.get('min_child_samples', 50)),
            subsample=float(data.get('subsample', 0.8)),
        )
        result = start_mlbase_test_task(
            current_app._get_current_object(),
            project_root=project_root,
            run_kwargs=run_kwargs,
        )
        if not result.get('success'):
            return jsonify(result), 409
        return jsonify(result)
    except ValueError as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/mlbase_test/status', methods=['GET'])
def mlbase_test_status():
    """轮询 MLBase Test 评估任务。"""
    since = request.args.get('log_since', 0, type=int)
    project_root = current_app.config['PROJECT_ROOT']
    status = get_mlbase_test_status(log_since=since, project_root=project_root)
    return jsonify({'success': True, **status})


@data_tool_bp.route('/mlbase_test/result', methods=['GET'])
def mlbase_test_result():
    """读取已保存的 mlbase_test_metrics.json。"""
    project_root = current_app.config['PROJECT_ROOT']
    metrics = load_mlbase_test_metrics(project_root)
    if not metrics:
        return jsonify({'success': False, 'error': '尚无 Test 评估结果'}), 404
    return jsonify({'success': True, 'metrics': metrics})


@data_tool_bp.route('/download_predictions')
def download_predictions():
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'pu_learning')
    return send_from_directory(results_path, "test_predictions.csv", as_attachment=True)

@data_tool_bp.route('/download_results')
def download_results():
    results_path = os.path.join(
        current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection'
    )
    filename = request.args.get('file', FEATURE_CANDIDATES_CSV)
    if filename not in FEATURE_SELECTION_DOWNLOAD_WHITELIST:
        return jsonify({'success': False, 'error': '不允许下载该文件'}), 400
    file_path = os.path.join(results_path, filename)
    if not os.path.isfile(file_path):
        if filename == FEATURE_CANDIDATES_CSV and os.path.isfile(
            os.path.join(results_path, 'feature_rank_comparison.csv')
        ):
            filename = 'feature_rank_comparison.csv'
        else:
            return jsonify({'success': False, 'error': '文件不存在'}), 404
    return send_from_directory(results_path, filename, as_attachment=True)


@data_tool_bp.route('/run_rfecv_reselection', methods=['POST'])
def run_rfecv_reselection():
    """RFECV 重选 Top 特征（后台任务）。"""
    try:
        project_root = current_app.config['PROJECT_ROOT']
        output_dir = os.path.join(project_root, 'data', 'results', 'feature_selection')
        cand_path = os.path.join(output_dir, FEATURE_CANDIDATES_CSV)
        if not os.path.isfile(cand_path):
            return jsonify({'success': False, 'error': '请先运行特征筛选'}), 400

        label_col = resolve_label_col_from_pu_session(project_root)
        dataset_type = resolve_dataset_type_from_pu_session(
            project_root, (request.json or {}).get('dataset_type', 'split')
        )
        result = start_rfecv_task(
            current_app._get_current_object(),
            project_root=project_root,
            label_col=label_col,
            dataset_type=dataset_type,
            output_dir=output_dir,
        )
        if not result.get('success'):
            return jsonify(result), 409
        return jsonify(result)
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/rfecv/status', methods=['GET'])
def rfecv_status():
    """轮询 RFECV 后台任务。"""
    since = request.args.get('log_since', 0, type=int)
    project_root = current_app.config['PROJECT_ROOT']
    status = get_rfecv_status(log_since=since, project_root=project_root)
    return jsonify({'success': True, **status})

@data_tool_bp.route('/get_full_results')
def get_full_results():
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'pu_learning', 'test_predictions.csv')
    if os.path.exists(results_path):
        df = _load_csv_robust(results_path)
        df_sample = df.head(100)
        df_sample = df_sample.fillna(value=np.nan)
        result_dict = df_sample.to_dict('records')
        for record in result_dict:
            for key, value in record.items():
                if isinstance(value, float) and np.isnan(value):
                    record[key] = None
        return jsonify(result_dict)
    else:
        return jsonify({'error': '预测结果文件未找到'}), 404

@data_tool_bp.route('/get_results_data')
def get_results_data():
    """兼容旧前端：返回 feature_candidates 记录列表。"""
    output_dir = os.path.join(
        current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection'
    )
    path = os.path.join(output_dir, FEATURE_CANDIDATES_CSV)
    legacy = os.path.join(output_dir, 'feature_rank_comparison.csv')
    if os.path.exists(path):
        df = _load_csv_robust(path)
    elif os.path.exists(legacy):
        df = _load_csv_robust(legacy)
    else:
        return jsonify({'error': '结果文件未找到'}), 404
    df_sample = df.head(500).astype(object).where(pd.notnull(df), None)
    return jsonify(df_sample.to_dict('records'))


@data_tool_bp.route('/get_top_features', methods=['GET'])
def get_top_features():
    """获取已确认 Top 特征列表（优先 top_features.csv）。"""
    try:
        output_dir = os.path.join(
            current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection'
        )
        top_path = os.path.join(output_dir, TOP_FEATURES_CSV)
        cand_path = os.path.join(output_dir, FEATURE_CANDIDATES_CSV)

        def _feature_rows_from_df(df: pd.DataFrame) -> list:
            if 'feature_en' not in df.columns:
                raise ValueError('特征表缺少 feature_en 列')
            has_zh = 'feature_zh' in df.columns
            rows = []
            for _, row in df.iterrows():
                name = str(row['feature_en']).strip()
                if not name:
                    continue
                if has_zh and pd.notna(row['feature_zh']) and str(row['feature_zh']).strip():
                    zh = str(row['feature_zh']).strip()
                else:
                    zh = name
                rows.append({'name': name, 'chinese_name': zh})
            return rows

        source = None
        if os.path.isfile(top_path):
            df = _load_csv_robust(top_path)
            top_features = _feature_rows_from_df(df)
            source = TOP_FEATURES_CSV
        elif os.path.isfile(cand_path):
            df = _load_csv_robust(cand_path)
            if 'selected' in df.columns:
                selected_mask = df['selected'].apply(
                    lambda v: str(v).strip().lower() in ('true', '1', 'yes')
                    if not isinstance(v, (bool, np.bool_))
                    else bool(v)
                )
                sel = df[selected_mask]
            else:
                sel = df.iloc[0:0]
            if sel.empty:
                return jsonify({
                    'success': False,
                    'error': (
                        '未找到已确认特征。请先在「特征工程」页完成特征筛选，'
                        '勾选特征后点击「确认特征」生成 top_features.csv'
                    ),
                }), 404
            top_features = _feature_rows_from_df(sel)
            source = FEATURE_CANDIDATES_CSV
        else:
            return jsonify({
                'success': False,
                'error': (
                    '未找到特征选择结果。请先在「特征工程」页运行特征筛选；'
                    f'期望文件：{TOP_FEATURES_CSV} 或 {FEATURE_CANDIDATES_CSV}'
                ),
            }), 404

        if not top_features:
            return jsonify({
                'success': False,
                'error': f'{source or TOP_FEATURES_CSV} 中无有效特征行',
            }), 404

        return jsonify({
            'success': True,
            'features': top_features,
            'count': len(top_features),
            'source': source,
        })
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500

# AI 参数智能推荐 API
@data_tool_bp.route('/optimize_pu_params', methods=['POST'])
def optimize_pu_params():
    """调用大模型 API，结合数据集分析报告与 PU_bagging 算法推荐参数"""
    try:
        data = request.json or {}
        api_key = data.get('api_key') or ''
        base_url = data.get('base_url') or os.environ.get('OPENAI_BASE_URL')
        model = data.get('model') or os.environ.get('PU_PARAM_LLM_MODEL', 'gpt-4o-mini')

        project_root = current_app.config['PROJECT_ROOT']
        log_folder = current_app.config.get('LOG_FOLDER', os.path.join(project_root, 'logs'))

        result = optimize_pu_params_with_llm(
            project_root=project_root,
            api_key=api_key,
            base_url=base_url,
            model=model,
            log_folder=log_folder,
        )

        results_dir = os.path.join(project_root, 'data', 'results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, 'pu_params_recommendation.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        return jsonify({
            'success': True,
            'params': result['params'],
            'reasoning': result.get('reasoning', ''),
            'llm_response': result.get('llm_response', ''),
            'model': result.get('model', model),
            'saved_path': save_path,
        })

    except (ConnectionError, RuntimeError, FileNotFoundError, ValueError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500

@data_tool_bp.route('/optimize_fe_params', methods=['POST'])
def optimize_fe_params():
    """LLM 推荐特征选择流水线参数。"""
    try:
        data = request.json or {}
        api_key = data.get('api_key') or ''
        base_url = data.get('base_url') or os.environ.get('OPENAI_BASE_URL')
        model = data.get('model') or os.environ.get('PU_PARAM_LLM_MODEL', 'gpt-4o-mini')
        project_root = current_app.config['PROJECT_ROOT']
        result = optimize_fe_params_with_llm(
            project_root,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        return jsonify({'success': True, **result})
    except (ConnectionError, RuntimeError, FileNotFoundError, ValueError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/optimize_mlbase_params', methods=['POST'])
def optimize_mlbase_params():
    """LLM 推荐 MLBase 单次训练参数。"""
    try:
        data = request.json or {}
        api_key = data.get('api_key') or ''
        base_url = data.get('base_url') or os.environ.get('OPENAI_BASE_URL')
        model = data.get('model') or os.environ.get('PU_PARAM_LLM_MODEL', 'gpt-4o-mini')
        project_root = current_app.config['PROJECT_ROOT']
        result = optimize_mlbase_params_with_llm(
            project_root,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        return jsonify({'success': True, **result})
    except (ConnectionError, RuntimeError, FileNotFoundError, ValueError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        log_exception(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_tool_bp.route('/optimize_feature_selection_params', methods=['POST'])
def optimize_feature_selection_params():
    """兼容旧路由：等同于 optimize_fe_params。"""
    return optimize_fe_params()

# 兼容旧的上传接口
@data_tool_bp.route('/upload', methods=['POST'])
def upload_file():
    return upload_full()
