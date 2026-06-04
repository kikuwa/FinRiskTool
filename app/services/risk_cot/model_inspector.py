import logging
import re
import time
import requests
import json
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

class ModelScorer:
    """
    基于模型（DeepSeek）的SFT语料质量评估
    """
    
    DEFAULT_SYSTEM_PROMPT = """你是一个SFT语料质量评估专家，任务是从原始数据集中筛选出高质量样本，特别关注思维链合成数据的难度和质量。

[质量检查维度]
1. 基础质量（0-3分）：
   - 准确性：需正确解决问题，无事实性错误或逻辑矛盾
   - 信息量：避免冗余回复或过于简略的答案，内容充实有价值
   - 语法规范：无拼写错误、标点误用或语义不通顺问题
   - 格式规范：符合对话格式要求，角色分明

2. 思维链难度评估（0-5分）：
   - 推理复杂度：需要多步推理、逻辑跳跃或复杂分析的难度越高
   - 知识深度：涉及专业知识、跨领域知识或深度理解的难度
   - 创造性：需要创造性思维、新颖视角或独特解决方案
   - 上下文理解：对复杂上下文、隐含信息或微妙语义的理解要求
   - 问题复杂度：原始问题的复杂程度和挑战性

3. 合成质量评估（0-2分）：
   - 自然流畅度：思维链是否自然连贯，无明显人工拼接痕迹
   - 逻辑一致性：推理过程逻辑严密，前后一致无矛盾
   - 步骤完整性：关键推理步骤完整，无跳跃或缺失
   - 可解释性：推理过程清晰可理解，便于人类follow

[评估步骤]
1. 初步筛选：检查敏感词、广告、无效占位符、明显低质量内容（直接0分）
2. 基础质量评分：按准确性、信息量、语法规范打分（0-3分）
3. 难度评估：评估思维链合成难度，难度越高得分越低（0-5分，越高越难）
4. 合成质量：评估思维链的自然性、逻辑性和完整性（0-2分）

[最终得分计算]
最终得分 = 基础质量分 × (1 - 难度分/10) + 合成质量分
说明：难度越高（难度分越大），对基础质量的折扣越大，体现合成难度对质量的负面影响

[格式要求]
---
语料得分：[最终得分（保留2位小数）]
---"""

    def __init__(self, api_key: str = None, api_base: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        self.default_api_key = api_key
        self.default_api_base = api_base
        self.default_model = model
        self.max_retries = 3
        self.timeout = 30

    def _user_prompt(self, instruction, input_text, output):
        return f"""请评估以下语料质量：

Instruction: {instruction}

Input: {input_text}

Output: {output}

请按照要求进行评分并返回格式化的结果。"""

    def _extract_score(self, content):
        """从响应中提取分数"""
        if not content:
            return None
        
        # 主模式
        match = re.search(r"语料得分：[^\d]*(\d+(?:\.\d+)?)", content)
        if match:
            return float(match.group(1))
        
        # 回退：取最后一个数字
        matches = re.findall(r"(\d+(?:\.\d+)?)", content)
        return float(matches[-1]) if matches else None

    def _api_request(self, messages, api_key, api_base, model):
        """发送API请求"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 100,
        }
        
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                logger.error(f"API错误 (第{attempt}次尝试): {resp.status_code} - {resp.text}")
            except Exception as e:
                logger.error(f"请求异常 (第{attempt}次尝试): {e}")
            
            if attempt < self.max_retries:
                time.sleep(2 ** (attempt - 1))
        return None

    def score_single(self, instruction, input_text, output, api_key=None, api_base=None, model_name=None, system_prompt=None):
        """对单条样本评分"""
        if api_key is not None:
            target_api_key = api_key
        else:
            target_api_key = self.default_api_key or ''
        target_api_base = api_base or self.default_api_base
        target_model = model_name or self.default_model

        if not (instruction or output): # allow empty input
             return 0.0, "Empty content"

        messages = [
            {"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(instruction, input_text, output)},
        ]
        content = self._api_request(messages, target_api_key, target_api_base, target_model)
        score = self._extract_score(content)
        return (score if score is not None else 0.0), content
