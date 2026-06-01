import logging
import re
from typing import Any, Dict, List, Optional

import openai
import pandas as pd

logger = logging.getLogger(__name__)


class PromptEngine:
    """
    Prompt 工程：提供参考 instruction 模板，按 CSV 行与已确认 Top 特征填充 Alpaca 数据。
    instruction 由用户在界面编辑；input 为每行特征构成的【企业指标数据报告】。
    """

    BASE_INSTRUCTION_TEMPLATE = """【角色】
您是一位具备金融风控专业知识的智能分析师，擅长结合企业定性及定量的指标数据对企业违约风险进行评估。

【任务指令】
请依据下方的【风险分析框架】和输入中提供的【企业指标数据报告】，按以下步骤对该企业信用风险进行综合判断：
1. 依据【风险分析框架】对企业经营、债务、信用等维度进行分析；
2. 通过指标间的交叉验证，判断是否存在框架未覆盖的风险点；
3. 综合评估企业在未来3个月内的违约风险。

【风险分析框架】
（1）企业经营稳定性：结合企业属性、行业及收支流水，判断经营是否良好及具备偿债能力。
（2）企业债务和流动性：结合短借长投、债务结构，判断是否过度举债。
（3）其它风险信号：关注机器学习评分、行内评级及外部征信数据。

【强制约束】
1. 必须建立双向验证机制：当关键信号与指标数据矛盾时，需启动二次核查；
2. 严格区分事实指标与推测结论。

【输出要求】
根据以上分析，如果认为该企业在未来3个月内不会违约，请输出“否”；如果存在违约风险，请输出“是”。请直接输出结果，不要包含其他分析过程。

【特征字段列表】
（导入已确认特征后自动填充）"""

    @classmethod
    def build_feature_list_section(cls, features: List[Dict[str, str]]) -> str:
        """根据 Top 特征生成可写入 instruction 的特征列表段落。"""
        if not features:
            return '【特征字段列表】\n（导入已确认特征后自动填充）'
        lines = [f'【特征字段列表（已确认 {len(features)} 个）】']
        for idx, feat in enumerate(features, start=1):
            name = str(feat.get('name') or '').strip()
            if not name:
                continue
            zh = str(feat.get('chinese_name') or name).strip()
            lines.append(f'{idx}. {zh} {{{name}}}')
        return '\n'.join(lines)

    @classmethod
    def build_data_report(
        cls,
        row: pd.Series,
        features: Optional[List[Dict[str, str]]] = None,
        *,
        label_col: str = 'label',
    ) -> str:
        """从 CSV 行构建 Alpaca input：【企业指标数据报告】。"""
        lines = ['【企业指标数据报告】']
        if features:
            for feat in features:
                name = str(feat.get('name') or '').strip()
                if not name or name == label_col:
                    continue
                zh = str(feat.get('chinese_name') or name).strip()
                val = row.get(name, '未知')
                if pd.isna(val):
                    val = '未知'
                lines.append(f'- {zh}（{name}）：{val}')
        else:
            for col in row.index:
                if col == label_col:
                    continue
                val = row[col]
                if pd.isna(val):
                    val = '未知'
                lines.append(f'- {col}：{val}')
        return '\n'.join(lines)

    @classmethod
    def generate_template_from_llm(
        cls,
        features: List[str],
        api_key: str = None,
        base_url: str = None,
        model: str = 'gpt-3.5-turbo',
    ) -> str:
        """调用 LLM 优化 Prompt 模板（失败时返回参考模板）。"""
        if not api_key:
            return cls.BASE_INSTRUCTION_TEMPLATE

        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            feature_list_str = '\n'.join([f'- {f}' for f in features])

            system_prompt = '你是一个精通提示词工程（Prompt Engineering）和金融风控的专家。'
            user_prompt = f"""
请根据以下【基础模板】和【特征字段列表】，重写并优化一个用于企业违约风险评估的 Prompt 模板。

【任务要求】
1. 保持基础模板的整体结构（角色、任务指令、风险分析框架、强制约束、输出要求）。
2. 重点优化【风险分析框架】部分：请根据特征字段的业务含义，将它们合理地嵌入到对应的分析维度中。
3. 嵌入槽位时，请严格使用 Python 格式化字符串的语法，即 `{{字段名}}`，且字段名须来自下方特征列表的英文名。
4. 确保所有重要的特征字段都被利用到。
5. 在模板末尾保留【企业指标数据报告】与【特征字段列表】占位说明；特征列表放在企业指标数据报告之后。
6. 直接返回优化后的模板内容，不要包含任何解释性文字或 Markdown 代码块。

【基础模板】
{cls.BASE_INSTRUCTION_TEMPLATE}

【特征字段列表】
{feature_list_str}
"""
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=0.7,
                max_tokens=1500,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith('```'):
                content = content.replace('```python', '').replace('```', '').strip()
            return content
        except Exception as exc:
            logger.error('LLM 生成模板失败: %s', exc)
            return cls.BASE_INSTRUCTION_TEMPLATE

    @classmethod
    def process_data(
        cls,
        df: pd.DataFrame,
        instruction_template: str = None,
        features: Optional[List[Dict[str, str]]] = None,
        label_col: str = 'label',
    ) -> List[Dict[str, Any]]:
        """将 CSV 转为 Alpaca 列表。"""
        if instruction_template is None:
            instruction_template = cls.BASE_INSTRUCTION_TEMPLATE

        df = df.copy().fillna('未知')
        result: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item = cls._build_alpaca_item(
                row,
                instruction_template,
                features=features,
                label_col=label_col,
            )
            if item:
                result.append(item)
        return result

    @staticmethod
    def _cell_value(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return '未知'
        if pd.isna(value):
            return '未知'
        return str(value)

    @classmethod
    def _safe_format_template(cls, template: str, row: pd.Series) -> str:
        """仅替换模板中出现的 `{列名}`，缺失列填「未知」。"""
        placeholders = set(re.findall(r'\{(\w+)\}', template))
        if not placeholders:
            return template
        data = {key: cls._cell_value(row.get(key)) for key in placeholders}
        return template.format(**data)

    @classmethod
    def _resolve_gt(cls, row: pd.Series, label_col: str = 'label') -> str:
        if label_col not in row.index:
            return ''
        raw = row.get(label_col)
        text = cls._cell_value(raw)
        if text == '未知':
            return ''
        if text in ('0', '0.0'):
            return '否'
        if text in ('1', '1.0'):
            return '是'
        return text

    @classmethod
    def _build_alpaca_item(
        cls,
        row: pd.Series,
        instruction_template: str,
        *,
        features: Optional[List[Dict[str, str]]] = None,
        label_col: str = 'label',
    ) -> Optional[Dict[str, Any]]:
        try:
            instruction_content = cls._safe_format_template(instruction_template, row)
            input_content = cls.build_data_report(row, features, label_col=label_col)
            gt_value = cls._resolve_gt(row, label_col)
            return {
                'instruction': instruction_content.strip(),
                'input': input_content.strip(),
                'output': '',
                'gt': gt_value,
            }
        except Exception as exc:
            logger.error('构建 Alpaca 数据失败: %s', exc, exc_info=True)
            return None
