
import json
import os
import logging
import threading
import time
from typing import Dict, List, Any
from .rule_base import RuleBase
from .model_inspector import ModelScorer

logger = logging.getLogger(__name__)

class InspectorEngine:
    """
    质检引擎服务：统一处理规则质检和模型质检
    """
    
    def __init__(self):
        self._stop_event = threading.Event()
        self._status = {
            "status": "idle", # idle, running, stopped, completed, error
            "total": 0,
            "processed": 0,
            "current_file": None,
            "inspection_type": None,
            "error": None
        }
        self._lock = threading.Lock()
        self.rule_base = RuleBase()
        self.model_scorer = ModelScorer()

    def get_status(self) -> Dict:
        """获取当前运行状态"""
        return self._status

    def stop(self):
        """停止任务"""
        self._stop_event.set()
        with self._lock:
            if self._status["status"] == "running":
                self._status["status"] = "stopped"

    def run(self, config: Dict):
        """
        运行质检任务
        config: {
            'input_file': str,
            'output_file': str,
            'type': 'rule' | 'model',
            # 模型质检参数
            'api_key': str,
            'api_base': str,
            'model': str,
            'system_prompt': str,
            # 规则质检参数
            'enabled_rules': List[str]
        }
        """
        self._stop_event.clear()
        
        with self._lock:
            self._status = {
                "status": "running",
                "total": 0,
                "processed": 0,
                "current_file": config.get('input_file'),
                "inspection_type": config.get('type', 'rule'),
                "error": None
            }

        try:
            self._execute_inspection(config)
            with self._lock:
                if self._status["status"] == "running":
                    self._status["status"] = "completed"
        except Exception as e:
            logger.error(f"质检任务异常: {e}", exc_info=True)
            with self._lock:
                self._status["status"] = "error"
                self._status["error"] = str(e)

    def _execute_inspection(self, config: Dict):
        input_file = config['input_file']
        output_file = config['output_file']
        inspection_type = config.get('type', 'rule')
        
        # 1. 加载数据
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
        
        total = len(data)
        with self._lock:
            self._status["total"] = total
            
        results = []
        
        # 2. 逐条处理
        for idx, item in enumerate(data):
            if self._stop_event.is_set():
                break
                
            processed_item = item.copy()
            
            try:
                if inspection_type == 'rule':
                    processed_item = self._run_rule_check(item, config)
                elif inspection_type == 'model':
                    processed_item = self._run_model_check(item, config)
                    # 模型质检限速
                    if (idx + 1) % 10 == 0:
                        time.sleep(1)
                else:
                    raise ValueError(f"未知的质检类型: {inspection_type}")
            except Exception as e:
                logger.error(f"处理第 {idx} 条数据出错: {e}")
                processed_item['error'] = str(e)
            
            results.append(processed_item)
            
            with self._lock:
                self._status["processed"] = idx + 1
        
        # 3. 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def _run_rule_check(self, item: Dict, config: Dict) -> Dict:
        """执行规则质检"""
        enabled_rules = config.get('enabled_rules', [])
        
        # 构造 RuleBase 需要的输入格式
        rule_input = {
            'meta_prompt': item.get('instruction', ''),
            'user': item.get('input', ''),
            'assistant': item.get('output', ''),
            'file_path': item.get('file_path', ''),
            'ref_answer': item.get('gt', None),
            'enabled_rules': enabled_rules if enabled_rules else None
        }
        
        rule_result = self.rule_base.run(rule_input)
        
        # 计算得分：只使用启用的规则
        rules_to_score = rule_result.get('enabled_rules', [])
        if not rules_to_score:
            rules_to_score = [
                k for k, v in rule_result.items() 
                if isinstance(v, bool) and k != 'warning'
            ]
        
        passed = sum(1 for k in rules_to_score if rule_result.get(k, False))
        total = len(rules_to_score)
        score = (passed / total * 10) if total > 0 else 0
        
        item['score'] = round(score, 2)
        item['rule_details'] = rule_result
        item['inspection_type'] = 'rule_based'
        return item

    def _run_model_check(self, item: Dict, config: Dict) -> Dict:
        """执行模型质检"""
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if not (instruction and output):
            item['score'] = 0.0
            return item

        api_key = config.get('api_key')
        api_base = config.get('api_base')
        model = config.get('model')
        system_prompt = config.get('system_prompt')
        
        score, reason = self.model_scorer.score_single(
            instruction, input_text, output,
            api_key=api_key,
            api_base=api_base,
            model_name=model,
            system_prompt=system_prompt
        )
        
        item['score'] = score if score is not None else 0.0
        item['model_reason'] = reason
        item['inspection_type'] = 'model_based'
        return item
