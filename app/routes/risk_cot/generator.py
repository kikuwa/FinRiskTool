from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.services.risk_cot.prompt_engine import PromptEngine
from app.services.data_core.feature_selection.pipeline import TOP_FEATURES_CSV
from app.services.data_core.shared.pu_session import resolve_label_col
import os
import uuid
import pandas as pd
import json
import numpy as np

generator_bp = Blueprint('generator', __name__, url_prefix='/api/generator')
ALPACA_ENGINE_MODE = 'top_features_input_v2'
ALPACA_SCHEMA_VERSION = '2.0'


def _load_csv_robust(path: str) -> pd.DataFrame:
    for enc in ('utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding='utf-8', errors='replace')


def _feature_rows_from_df(df: pd.DataFrame) -> list:
    if 'feature_en' not in df.columns:
        return []
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


def _load_top_features_for_prompt(project_root: str) -> list:
    """读取 top_features.csv 中的已确认特征（供 Alpaca 填充 input）。"""
    output_dir = os.path.join(project_root, 'data', 'results', 'feature_selection')
    top_path = os.path.join(output_dir, TOP_FEATURES_CSV)
    if not os.path.isfile(top_path):
        return []
    df = _load_csv_robust(top_path)
    return _feature_rows_from_df(df)

def clean_for_json(obj):
    """将对象中的 NaN, Inf, -Inf 转换为 None，以便 JSON 序列化"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj

@generator_bp.route('/check_train_data', methods=['GET'])
def check_train_data():
    train_files = ['train.csv', 'train_dataset.csv']
    data_folder = current_app.config['DATA_FOLDER']
    
    for display_name in train_files:
        file_path = os.path.join(data_folder, display_name)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                preview = df.head(5).where(pd.notnull(df), None).to_dict(orient='records')
                preview = clean_for_json(preview)
                return jsonify({
                    'status': 'success',
                    'exists': True,
                    'filename': display_name,
                    'display_name': display_name,
                    'preview': preview
                })
            except Exception as e:
                continue
    
    return jsonify({'status': 'success', 'exists': False})

@generator_bp.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    if file:
        try:
            filename = secure_filename(file.filename)
            # Unique prefix to avoid overwrites
            task_id = str(uuid.uuid4())[:8]
            filename = f"{task_id}_{filename}"
            
            save_path = os.path.join(current_app.config['DATA_FOLDER'], filename)
            file.save(save_path)
            
            # Try to read it to get preview
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(save_path)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(save_path)
            else:
                 # Clean up if invalid
                 os.remove(save_path)
                 return jsonify({'status': 'error', 'message': 'Unsupported file format. Please upload CSV or Excel.'}), 400
                 
            # Convert NaN to None for JSON compatibility
            preview = df.head(5).where(pd.notnull(df), None).to_dict(orient='records')
            # Clean NaN, Inf values for JSON serialization
            preview = clean_for_json(preview)
            
            return jsonify({
                'status': 'success',
                'message': '文件上传成功',
                'filename': filename,
                'preview': preview
            })
        except Exception as e:
             return jsonify({'status': 'error', 'message': f'Failed to process file: {str(e)}'}), 500

@generator_bp.route('/template/default', methods=['GET'])
def get_default_template():
    return jsonify({
        'status': 'success',
        'template': PromptEngine.BASE_INSTRUCTION_TEMPLATE
    })

@generator_bp.route('/template/generate', methods=['POST'])
def generate_template():
    try:
        api_key = request.json.get('api_key') or ''
        base_url = request.json.get('base_url')
        model = request.json.get('model', 'gpt-3.5-turbo')
        features = request.json.get('features', [])

        engine = PromptEngine()
        template = engine.generate_template_from_llm(
            features=features,
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        
        return jsonify({
            'status': 'success',
            'template': template
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@generator_bp.route('/alpaca', methods=['POST'])
def generate_alpaca():
    try:
        payload = request.json or {}
        source_file = payload.get('source_file')
        template = payload.get('template')
        features = payload.get('features') or []

        if not source_file:
            return jsonify({'status': 'error', 'message': 'Source file is required'}), 400

        project_root = current_app.config['PROJECT_ROOT']
        if not features:
            features = _load_top_features_for_prompt(project_root)
        if not features:
            return jsonify({
                'status': 'error',
                'message': '未找到已确认特征。请先在特征工程页确认特征（top_features.csv），再执行 Alpaca 转换。',
            }), 400

        # Full path check
        if not os.path.isabs(source_file):
            source_file = os.path.join(current_app.config['DATA_FOLDER'], source_file)

        if not os.path.exists(source_file):
            return jsonify({'status': 'error', 'message': 'Source file not found'}), 404

        output_filename = f"alpaca_{os.path.basename(source_file).replace('.csv', '')}.jsonl"
        output_path = os.path.join(current_app.config['DATA_FOLDER'], output_filename)

        df = _load_csv_robust(source_file)
        label_col = resolve_label_col(project_root)
        result_items = PromptEngine.process_data(
            df,
            instruction_template=template,
            features=features or None,
            label_col=label_col,
        )

        if not result_items:
            return jsonify({
                'status': 'error',
                'message': '未能生成任何 Alpaca 样本，请检查 CSV 与模板占位符',
            }), 400

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in result_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        return jsonify({
            'status': 'success',
            'message': f'成功转换 {len(result_items)} 条数据',
            'output_file': output_filename,
            'feature_count': len(features) if features else 0,
            'engine_mode': ALPACA_ENGINE_MODE,
            'schema_version': ALPACA_SCHEMA_VERSION,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
