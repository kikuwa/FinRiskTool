
import pandas as pd
import json
import logging
import os
from typing import List, Dict, Optional
import openai
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptEngine:
    """
    Prompt 工程服务：负责将原始数据转换为 Alpaca 格式的指令数据集
    """
    
    # 基础模板
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

【企业指标数据报告】
（动态填充部分）
"""

    FALLBACK_INSTRUCTION_TEMPLATE = """【角色】
您是一位具备金融风控专业知识的智能分析师，擅长结合企业定性及定量的指标数据对企业违约风险进行评估。

【任务指令】
请依据下方的【风险分析框架】和输入中提供的【企业指标数据报告】，按以下步骤对该企业信用风险进行综合判断：
1. 依据【风险分析框架】对企业经营、债务、信用等维度进行分析；
2. 通过指标间的交叉验证，判断是否存在框架未覆盖的风险点；
3. 综合评估企业在未来3个月内的违约风险。

【风险分析框架】
（1）企业经营稳定性：结合企业属性（{nature}）、行业（{industry_cd_desc}）及收支流水，判断经营是否良好及具备偿债能力。关注近24个月营业收入（{corp_oper_income_24m}）与支出（{corp_oper_expns_24m}），以及经营净收益的变动（近12个月：{corp_oper_diff_12m}，近24个月：{corp_oper_diff_24m}）。参考纳税等级（{taxes_level_cd}）。
（2）企业债务和流动性：结合短借长投、债务结构，判断是否过度举债。分析当前金融资产余额（{corp_curr_fin_asset_bal}）及其变动（{corp_curr_fin_asset_bal_diff}），以及近3个月平均存款余额（{total_avg_depst_bal_3m}）。关注融资情况（融资机构数：{n_ph_num4}，融资余额：{n_ph_amt4}，本行融资：{n_ph_amt2}）及授信使用（授信额度：{creditamt}，已用：{amtused}）。
（3）其它风险信号：关注贷后检查结果（{check_result}）、风险标识（{risk_flag}）、重要股东风险（{reserve_8}）及押品跌幅（{down30}）。检查逾期情况（近3个月逾期：{rep_overdue_cnt_3m}，近24个月逾期：{rep_overdue_cnt_24m}，当前逾期金额：{corp_curr_lab_amt}）。参考模型评分（{threshold}）和外部规则命中（{n_u0343_hit}, {n_ph_amt1_hit}, {n_ph_amt2_hit}）。

【强制约束】
1. 必须建立双向验证机制：当关键信号与指标数据矛盾时，需启动二次核查；
2. 严格区分事实指标与推测结论。

【输出要求】
根据以上分析，如果认为该企业在未来3个月内不会违约，请输出“否”；如果存在违约风险，请输出“是”。请直接输出结果，不要包含其他分析过程。

【企业指标数据报告】
企业名称：{entityname}
行业：{industry_cd_desc}
分行：{branchname}
首次授信年份：{firstcreditx_year}
（详细数据见Input部分）"""

    INPUT_DATA_TEMPLATE = """【企业指标数据报告】
基本信息：
- 企业名称：{entityname}
- 企业性质：{nature}
- 首次授信年份：{firstcreditx_year}
- 所属行业：{industry_cd_desc}
- 所属分行：{branchname}

信用评级与风险：
- 信用评级：{cred_level_desc}
- 风险阈值：{threshold}
- 风险标识：{risk_flag}
- 统计算法结果：{tj_result}
- 贷后检查结果：{check_result}

经营状况：
- 近24个月营业收入：{corp_oper_income_24m}
- 近24个月营业支出：{corp_oper_expns_24m}
- 近12个月经营净收益：{corp_oper_diff_12m}
- 近24个月经营净收益：{corp_oper_diff_24m}
- 纳税等级：{taxes_level_cd}

资产与负债：
- 当前金融资产余额：{corp_curr_fin_asset_bal}
- 金融资产余额变动：{corp_curr_fin_asset_bal_diff}
- 近3个月平均存款余额：{total_avg_depst_bal_3m}
- 融资机构数量：{n_ph_num4}
- 融资余额：{n_ph_amt4}
- 融资金额：{n_ph_amt2}
- 授信额度：{creditamt}
- 已用额度：{amtused}

逾期与违约：
- 未结清比例：{unpayoff_mg_ratio}
- 近3个月还款逾期次数：{rep_overdue_cnt_3m}
- 近24个月还款逾期次数：{rep_overdue_cnt_24m}
- 当前逾期金额：{corp_curr_lab_amt}
- 本金/利息逾期标识：{flag_8}

规则命中：
- 规则U0343命中情况：{n_u0343_hit}
- 规则PH_AMT1命中情况：{n_ph_amt1_hit}
- 规则PH_AMT2命中情况：{n_ph_amt2_hit}

信用卡及其他：
- 信用卡额度：{rep_crdt_card_limit}
- 近6个月平均信用卡额度：{rep_crdt_card_avg_limit_6m}
- 信用卡已用额度：{rep_crdt_card_used_limit}
- 信用卡预支使用率：{rep_crdt_card_precent}
- 重要股东风险：{reserve_8}
- 下调30%标识：{down30}"""

    @classmethod
    def generate_template_from_llm(cls, features: List[str], api_key: str = None, base_url: str = None, model: str = "gpt-3.5-turbo") -> str:
        """调用 LLM 生成 Prompt 模板"""
        if not api_key:
            return cls.FALLBACK_INSTRUCTION_TEMPLATE

        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            feature_list_str = "\n".join([f"- {f}" for f in features])
            
            system_prompt = "你是一个精通提示词工程（Prompt Engineering）和金融风控的专家。"
            user_prompt = f"""
请根据以下【基础模板】和【特征字段列表】，重写并优化一个用于企业违约风险评估的 Prompt 模板。

【任务要求】
1. 保持基础模板的整体结构（角色、任务指令、风险分析框架、强制约束、输出要求）。
2. 重点优化【风险分析框架】部分：请根据特征字段的业务含义，将它们合理地嵌入到对应的分析维度中。
3. 嵌入槽位时，请严格使用Python格式化字符串的语法，即 `{{字段名}}`。
4. 确保所有重要的特征字段都被利用到。
5. 在模板末尾保留【企业指标数据报告】部分。
6. 直接返回优化后的模板内容，不要包含任何解释性文字或Markdown代码块。

【基础模板】
{cls.BASE_INSTRUCTION_TEMPLATE}

【特征字段列表】
{feature_list_str}
"""
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            content = response.choices[0].message.content.strip()
            # 清理
            if content.startswith("```"):
                content = content.replace("```python", "").replace("```", "").strip()
            return content
        except Exception as e:
            logger.error(f"LLM 生成模板失败: {e}")
            return cls.FALLBACK_INSTRUCTION_TEMPLATE

    @classmethod
    def process_data(cls, df: pd.DataFrame, instruction_template: str = None) -> List[Dict]:
        """处理数据框，返回 Alpaca 格式列表"""
        if instruction_template is None:
            instruction_template = cls.FALLBACK_INSTRUCTION_TEMPLATE
            
        # 预处理
        df = cls._preprocess_dataframe(df)
        
        result = []
        for _, row in df.iterrows():
            item = cls._build_alpaca_item(row, instruction_template)
            if item:
                result.append(item)
        return result

    @staticmethod
    def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """向量化预处理"""
        if 'check_result' in df.columns:
            df['check_result'] = df['check_result'].apply(PromptEngine._clean_check_result)
        df = df.fillna("未知")
        return df

    @staticmethod
    def _clean_check_result(text: Optional[str]) -> str:
        if pd.isna(text) or text is None:
            return "无"
        try:
            records = str(text).split("\n\n")
            parsed = []
            for r in records:
                r = r.strip()
                if not r: continue
                parts = r.split(" ")
                if len(parts) >= 2:
                    parsed.append((parts[0], " ".join(parts[1:])))
                else:
                    parsed.append(("0000-00-00", r))
            parsed.sort(key=lambda x: x[0], reverse=True)
            return "\n".join([f"[{t}] {c}" for t, c in parsed[:2]])
        except:
            return str(text)

    @staticmethod
    def _format_percentage(row, limit_col, used_col):
        try:
            limit = pd.to_numeric(row.get(limit_col, 0), errors='coerce')
            used = pd.to_numeric(row.get(used_col, 0), errors='coerce')
            if limit > 0:
                return f"{used / limit:.2%}"
            return "0.00%"
        except:
            return "0.00%"

    @classmethod
    def _build_alpaca_item(cls, row_series: pd.Series, instruction_template: str) -> Optional[Dict]:
        data_dict = row_series.to_dict()
        data_dict['rep_crdt_card_precent'] = cls._format_percentage(
            row_series, 'rep_crdt_card_limit', 'rep_crdt_card_used_limit'
        )
        
        try:
            # 检查模板所需的字段是否都存在
            template_vars = set()
            for template in [cls.INPUT_DATA_TEMPLATE, instruction_template]:
                import re
                vars_found = re.findall(r'\{(\w+)\}', template)
                template_vars.update(vars_found)
            
            missing_vars = [v for v in template_vars if v not in data_dict]
            if missing_vars:
                import logging
                logging.warning(f"数据缺少模板所需字段: {missing_vars}")
                # 用"未知"填充缺失字段
                for var in missing_vars:
                    data_dict[var] = "未知"
            
            input_content = cls.INPUT_DATA_TEMPLATE.format(**data_dict)
            instruction_content = instruction_template.format(**data_dict)
            
            gt_value = str(row_series.get('label', ''))
            if not gt_value or gt_value == 'nan':
                gt_value = "是" if row_series.get('is_risky') else "否"

            return {
                "instruction": instruction_content.strip(),
                "input": input_content.strip(),
                "output": "",
                "gt": gt_value
            }
        except Exception as e:
            import logging
            logging.error(f"构建 Alpaca 数据失败: {e}", exc_info=True)
            return None
