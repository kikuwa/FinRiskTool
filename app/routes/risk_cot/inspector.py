from flask import Blueprint, request, jsonify, Response, current_app
from app.services.risk_cot import inspector_engine
import threading
import os
import json
import time

from werkzeug.utils import secure_filename

inspector_bp = Blueprint('inspector', __name__, url_prefix='/api/inspector')

@inspector_bp.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400
            
        if file:
            filename = secure_filename(file.filename)
            # Add timestamp to avoid collisions
            timestamp = int(time.time())
            filename = f"{timestamp}_{filename}"
            
            # Save to DATA_FOLDER as it is an input for inspection
            save_path = os.path.join(current_app.config['DATA_FOLDER'], filename)
            file.save(save_path)
            
            return jsonify({
                'status': 'success', 
                'message': 'File uploaded successfully',
                'filename': filename
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@inspector_bp.route('/run', methods=['POST'])
def run_inspection():
    try:
        data = request.json
        input_file = data.get('input_file')
        
        if not input_file:
            return jsonify({'status': 'error', 'message': 'Missing input_file'}), 400
            
        # Handle relative paths
        if not os.path.isabs(input_file):
            # Try finding in RESULTS_FOLDER (from inference) or DATA_FOLDER
            path_in_results = os.path.join(current_app.config['RESULTS_FOLDER'], input_file)
            path_in_data = os.path.join(current_app.config['DATA_FOLDER'], input_file)
            
            if os.path.exists(path_in_results):
                input_file = path_in_results
            elif os.path.exists(path_in_data):
                input_file = path_in_data
            else:
                return jsonify({'status': 'error', 'message': 'Input file not found'}), 404
            
        # Generate output filename - properly handle edge cases
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        # Clean up any trailing special characters
        base_name = base_name.rstrip('-_. ')
        inspection_type = data.get('type', 'rule')
        output_file = os.path.join(current_app.config['RESULTS_FOLDER'], f"{base_name}_{inspection_type}_inspected.jsonl")
        
        config = {
            'input_file': input_file,
            'output_file': output_file,
            'type': inspection_type,
            'enabled_rules': data.get('enabled_rules', []),
            # Model params
            'api_key': data.get('api_key'),
            'api_base': data.get('api_base'),
            'model': data.get('model'),
            'system_prompt': data.get('system_prompt')
        }
        
        # Check if already running
        status = inspector_engine.get_status()
        if status['status'] == 'running':
             return jsonify({'status': 'error', 'message': 'Task is already running'}), 409
        
        # Start in background thread
        thread = threading.Thread(target=inspector_engine.run, args=(config,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success', 
            'message': 'Inspection task started',
            'output_file': output_file
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@inspector_bp.route('/stop', methods=['POST'])
def stop_inspection():
    try:
        inspector_engine.stop()
        return jsonify({'status': 'success', 'message': 'Stop signal sent'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@inspector_bp.route('/status')
def stream_status():
    def generate():
        while True:
            status = inspector_engine.get_status()
            yield f"data: {json.dumps(status)}\n\n"
            if status['status'] in ['completed', 'stopped', 'error']:
                break
            time.sleep(1)
    
    return Response(generate(), mimetype='text/event-stream')

@inspector_bp.route('/results', methods=['GET'])
def get_results():
    try:
        file_path = request.args.get('file_path')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        if not file_path:
             return jsonify({'status': 'error', 'message': 'Missing file_path'}), 400

        # Handle relative paths (security check could be added here)
        if not os.path.isabs(file_path):
             file_path = os.path.join(current_app.config['RESULTS_FOLDER'], file_path)
             
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
            
        data = []
        total_items = 0
        
        # Read all lines (for simplicity, could be optimized for large files)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_items = len(lines)
            
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            
            for line in lines[start_idx:end_idx]:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
        
        # Calculate stats
        pass_count = 0
        avg_score = 0
        if data:
             # This stats is only for current page, ideal would be for whole file
             # For whole file stats, we'd need to parse all lines which might be slow
             # Let's do a quick scan if file is not too huge
             pass 

        return jsonify({
            'status': 'success',
            'data': data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_items,
                'total_pages': (total_items + per_page - 1) // per_page
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
