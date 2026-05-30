import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ========== 预编译正则表达式（提升性能） ==========
_TRUNCATED_PATTERNS = [
    re.compile(r'\.\.\.$'),
    re.compile(r'等\.$'),
    re.compile(r'\n\.\.\.$'),
    re.compile(r'\[\s*\.\.\.\s*\]$'),
    re.compile(r'请继续.*'),
    re.compile(r'续写.*')
]
_SPECIAL_TASK_KEYWORDS = re.compile(r"转写|抽取|摘要|报告|合同|纪要|提案|会议", re.IGNORECASE)
_INCOMPLETE_PATTERNS = [
    re.compile(r"请总结上述文本内容"),
    re.compile(r"翻译这个句子")
]
_ENGLISH_LETTER = re.compile(r'[a-zA-Z]')
_CHINESE_CHAR = re.compile(r'[\u4e00-\u9fff]')
_GPT_KEYWORDS = re.compile(
    r"gpt|chatgpt|gpt-4|cortana|gpt-3\.5|ernie|文心一言|ernie bot|文心大模型|"
    r"sparkdesk|讯飞|认知大模型|星火app|星火认知|通义千问|千问大模型|qwen|通义大模型|"
    r"chatglm|智谱|deepseek|深度求索", re.IGNORECASE
)
_BOXED_PATTERN = re.compile(r'\\boxed\{(.+?)\}', re.DOTALL)
_ANSWER_TAG_PATTERN = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL)
_THINK_PATTERN = re.compile(r'\<think\>(.*?)\</think\>', re.DOTALL)
_CRASHED_PATTERNS = [
    re.compile(r"[ΦДБЪΨ]"),
    re.compile(r"锟斤拷")
]
_BRACKET_PAIRS = [("【", "】"), ('《', '》'), ('(', ')'), ('[', ']'), ('{', '}'), ('“', '”')]
_EMOJI_LIKE = {';)', ':)', ';）', ':）', '：）', '：（', '：('}

# 代码关键词（用于豁免某些规则）
_CODE_KEYWORDS = {
    '代码', 'python', 'sql', 'java', 'cpp', 'bash', 'html', 'css', 'batch',
    'php', 'go', 'yaml', 'c++', 'shell', 'csharp', 'xml', 'c#', '```'
}
_CODE_FILE_INDICATORS = {'code', '编程能力', 'panguml', 'qa_62k_0420'}


class RuleBase:
    def __init__(self):
        self.warning_process: List[str] = []
        self.test_data: List[Dict[str, Any]] = []
        self.file_path: str = ''
        self.result: Dict[str, Any] = {}

    def is_not_string(self, text: str) -> bool:
        """判断是否为纯数字（忽略逗号和点）"""
        if not text:
            return False
        cleaned = re.sub(r'[,.]', '', text)
        return cleaned.isdigit()

    def _is_code_task(self) -> bool:
        """判断是否为代码类任务（基于路径或内容）"""
        lower_path = self.file_path.lower()
        if any(ind in lower_path for ind in _CODE_FILE_INDICATORS):
            return True
        if not self.test_data:
            return False
        sample = self.test_data[0]
        content = (sample['data'][0]['content'] + sample['data'][1]['content']).lower()
        return any(kw in content for kw in _CODE_KEYWORDS)

    def _is_math_task(self) -> bool:
        return "math" in self.file_path.lower()

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        meta_prompt = input_data.get('meta_prompt', '')
        user_content = str(input_data.get('user', ''))
        assistant_content = str(input_data.get('assistant', ''))
        self.file_path = input_data.get('file_path', '')
        ref_answer = input_data.get('ref_answer', None)
        enabled_rules = input_data.get('enabled_rules', None)

        all_rules = [
            'is_number', 'no_text_truncated', 'no_incomplete_content',
            'no_chinese_english_mix', 'no_repeat_content', 'no_unclose_paire',
            'no_repeat_pattern', 'no_crashed_str', 'no_chinese_English_space',
            'no_other_gpt_keywords', 'no_think', 'all_math_answer_equal',
            'fk_answer_exist', 'fk_answer_yes_or_no', 'fk_answer_equal'
        ]

        if enabled_rules:
            rules_to_check = [r for r in all_rules if r in enabled_rules]
        else:
            rules_to_check = all_rules

        self.result = {
            'is_number': True,
            'no_text_truncated': True,
            'no_incomplete_content': True,
            'no_chinese_english_mix': True,
            'no_repeat_content': True,
            'no_unclose_paire': True,
            'no_repeat_pattern': True,
            'no_crashed_str': True,
            'no_chinese_English_space': True,
            'no_other_gpt_keywords': True,
            'no_think': True,
            'all_math_answer_equal': True,
            'fk_answer_exist': True,
            'fk_answer_yes_or_no': True,
            'fk_answer_equal': True,
            'warning': '',
            'assistant_content': assistant_content,
            'enabled_rules': rules_to_check
        }
        self.warning_process = []

        # 构造内部数据结构
        self.test_data = [{
            "meta_prompt": [meta_prompt],
            "data": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "ref_answer": ref_answer
        }]

        # === 通用质量检查（任一失败即终止，除 think 外）===
        checks = [
            ('fk_answer_exist', self._check_fk_exist, 'fk_no_answer_exist'),
            ('is_number', self._check_is_number, 'number_data'),
            ('no_text_truncated', self._check_truncated, 'long_text'),
            ('no_incomplete_content', self._check_incomplete, 'incomplete_content'),
            ('no_chinese_english_mix', self._check_chinese_english_mix, 'no_chinese_english_mix'),
            ('no_repeat_content', self._check_repeat_content, 'repeat_content'),
            ('no_repeat_pattern', self._check_repeat_pattern, 'repeat_pattern'),
            ('no_unclose_paire', self._check_unclose_pair, 'unclose_paire'),
            ('no_crashed_str', self._check_crashed_str, 'crashed_str'),
            ('no_chinese_English_space', self._check_chinese_english_space, 'Chinese_English_space'),
            ('no_other_gpt_keywords', self._check_gpt_keywords, 'other_gpt_keywords'),
        ]

        for key, check_func, warning_msg in checks:
            if key not in rules_to_check:
                continue
            try:
                if not check_func():
                    self.result[key] = False
                    self.warning_process.append(warning_msg)
                    # 提前返回（think 检查不在这里）
                    self.result['warning'] = '; '.join(self.warning_process)
                    return self.result
            except Exception:
                # 安全兜底：视为检查失败
                self.result[key] = False
                self.warning_process.append(f"{warning_msg}_exception")
                self.result['warning'] = '; '.join(self.warning_process)
                return self.result

        # === 特殊检查：慢思考（不中断后续）===
        if 'no_think' in rules_to_check:
            try:
                if self._check_think():
                    self.result['no_think'] = False
                    self.warning_process.append('no_think')
            except Exception:
                self.result['no_think'] = False
                self.warning_process.append('no_think_exception')

        # === 数学答案检查（仅 math 文件且规则启用）===
        if self._is_math_task() and 'all_math_answer_equal' in rules_to_check:
            try:
                if not self._check_math_answer():
                    self.result['all_math_answer_equal'] = False
                    self.warning_process.append('not all_math_answer_equal')
                    self.result['warning'] = '; '.join(self.warning_process)
                    return self.result
            except Exception:
                self.result['all_math_answer_equal'] = False
                self.warning_process.append('math_check_exception')
                self.result['warning'] = '; '.join(self.warning_process)
                return self.result

        # === 金融风控检查（非 math/code 且规则启用）===
        is_fk_task = not self._is_math_task() and not self._is_code_task()
        if is_fk_task:
            if 'fk_answer_yes_or_no' in rules_to_check:
                try:
                    if not self._check_fk_yes_no():
                        self.result['fk_answer_yes_or_no'] = False
                        self.warning_process.append('fk_not_answer_yes_or_no')
                        self.result['warning'] = '; '.join(self.warning_process)
                        return self.result
                except Exception:
                    self.result['fk_answer_yes_or_no'] = False
                    self.warning_process.append('fk_yes_no_exception')
                    self.result['warning'] = '; '.join(self.warning_process)
                    return self.result

            if 'fk_answer_equal' in rules_to_check:
                try:
                    if not self._check_fk_answer_equal():
                        self.result['fk_answer_equal'] = False
                        self.warning_process.append('fk_answer_not_equal')
                        self.result['warning'] = '; '.join(self.warning_process)
                        return self.result
                except Exception:
                    self.result['fk_answer_equal'] = False
                    self.warning_process.append('fk_answer_equal_exception')
                    self.result['warning'] = '; '.join(self.warning_process)
                    return self.result

        self.result['warning'] = '; '.join(self.warning_process)
        return self.result

    # --- 封装检查函数 ---
    def _check_fk_exist(self) -> bool:
        content = self.test_data[0]['data'][1]['content']
        stripped = content.strip()
        if not stripped or stripped in {'<think>', '</think>', '<think>\n</think>'}:
            return False
        return True

    def _check_is_number(self) -> bool:
        user = self.test_data[0]['data'][0]['content']
        assistant = self.test_data[0]['data'][1]['content']
        return not (self.is_not_string(user) or self.is_not_string(assistant))

    def _check_truncated(self) -> bool:
        content = self.test_data[0]['data'][1]['content']
        if len(content) <= 500:
            return False
        if _SPECIAL_TASK_KEYWORDS.search(content):
            return False
        for pat in _TRUNCATED_PATTERNS:
            if pat.search(content.strip()):
                return True
        return False

    def _check_incomplete(self) -> bool:
        user = self.test_data[0]['data'][0]['content']
        return any(pat.search(user) for pat in _INCOMPLETE_PATTERNS)

    def _check_chinese_english_mix(self) -> bool:
        if self._is_code_task():
            return False
        user = self.test_data[0]['data'][0]['content']
        assistant = self.test_data[0]['data'][1]['content']
        return self.check_chinese_query_english_response(user, assistant) or \
               self.check_english_query_chinese_response(user, assistant)

    def _check_repeat_content(self) -> bool:
        return self.repeat_content_checking()

    def _check_repeat_pattern(self) -> bool:
        return self.repeat_pattern_check(self.test_data)

    def _check_unclose_pair(self) -> bool:
        return self.rule_pair_check(self.test_data)

    def _check_crashed_str(self) -> bool:
        return self.crashed_str_check(self.test_data)

    def _check_chinese_english_space(self) -> bool:
        if self._is_code_task():
            return False
        return self.Chinese_English_space_check(self.test_data)

    def _check_gpt_keywords(self) -> bool:
        return self.other_gpt_keywords(self.test_data)

    def _check_think(self) -> bool:
        return self.think_checking(self.test_data)

    def _check_math_answer(self) -> bool:
        return self.math_answer_checking(self.test_data)

    def _check_fk_yes_no(self) -> bool:
        return self.fk_yes_or_no_checking(self.test_data)

    def _check_fk_answer_equal(self) -> bool:
        return self.fk_answer_checking(self.test_data)

    # --- 原有辅助方法（保留并微调）---
    def english_data(self, text: str, p: float = 0.2) -> bool:
        if not text:
            return False
        text = re.sub(r'(?s)```.*?```', '', text)
        en_count = len(_ENGLISH_LETTER.findall(text))
        return en_count / max(len(text), 1) > p

    def check_chinese_query_english_response(self, query: str, response: str) -> bool:
        if not _CHINESE_CHAR.search(query):
            return False
        if any(kw in query for kw in ['英语：', '英文：', '翻译：', '英文', 'English']):
            return False
        return "let's" in response.lower() or self.english_data(response, 0.3)

    def check_english_query_chinese_response(self, query: str, response: str) -> bool:
        if _CHINESE_CHAR.search(query):
            return False
        if _CHINESE_CHAR.search(response) and '中文' not in query and 'chinese:' not in query.lower():
            return True
        return False

    def longest_dup_substring(self, s: str) -> str:
        if not s:
            return ""
        max_len = min(len(s) // 2, 100)
        for length in range(max_len, 1, -1):
            seen = set()
            for i in range(len(s) - length + 1):
                sub = s[i:i + length]
                if sub in seen:
                    return sub
                seen.add(sub)
        return ""

    def repeat_content_checking(self) -> bool:
        filepath = self.file_path
        try:
            for sample in self.test_data:
                for t in sample['data']:
                    if t['role'] == 'assistant':
                        content = str(t['content'])
                        dedup_result_eng = self.longest_dup_substring(content)
                        is_math_or_code = 'math' in filepath.lower() or 'code' in filepath.lower()

                        if len(dedup_result_eng) > 50 and not is_math_or_code:
                            return True
                        if len(dedup_result_eng) > 50 and is_math_or_code and content.count(dedup_result_eng) > 3:
                            return True

                        pt = content.replace(' ', '').replace('\n', '').replace('\t', '').replace('"', '').strip()
                        if not self.english_data(content.replace(' ', '')):
                            for c_ in set(pt):
                                if c_ * 40 in pt and c_ != '-':
                                    return True

                        pt_clean = re.sub(r'[\d\s\n\t"|]', '', pt)
                        if len(pt_clean) > 50:
                            if len(set(pt_clean)) / len(pt_clean) < 0.15:
                                promt_str = sample['data'][0]['content']
                                similar_pattern = "改写|近意|近似|相似|相同|同义"
                                if not re.search(similar_pattern, promt_str.lower()):
                                    return True
        except Exception:
            return True
        return False

    def repeat_pattern_check(self, test_data: List[Dict[str, Any]]) -> bool:
        for tmp_line in test_data:
            test_str = ''
            for sub_data in tmp_line['data']:
                if sub_data['role'] == 'assistant' and isinstance(sub_data['content'], str):
                    test_str += sub_data['content']
            pattern = r"(改写之后)|(助手回复)|([\u4e00-\u9fff]+-based)"
            if re.search(pattern, test_str):
                return True
        return False

    def rule_pair_check(self, test_data: List[Dict[str, Any]]) -> bool:
        try:
            for sub_data in test_data:
                for j in sub_data['data']:
                    if j['role'] == 'assistant':
                        content = str(j['content'])
                        remove_code = re.sub('(?s)```.*?```', '', content).lower()
                        if any(s in remove_code for s in _EMOJI_LIKE):
                            continue
                        for (left, right) in _BRACKET_PAIRS:
                            if remove_code.count(left) != remove_code.count(right):
                                return True
        except Exception:
            return True
        return False

    def crashed_str_check(self, test_data: List[Dict[str, Any]]) -> bool:
        try:
            for tmp_line in test_data:
                test_assi_str = ""
                test_user_str = ""
                for sub_data in tmp_line['data']:
                    if sub_data['role'] == 'assistant':
                        test_assi_str += sub_data['content']
                    elif sub_data['role'] == 'user':
                        test_user_str += sub_data['content']

                for pat in _CRASHED_PATTERNS:
                    if pat.search(test_assi_str) and not pat.search(test_user_str):
                        return True
        except Exception:
            return True
        return False

    def Chinese_English_space_check(self, test_data: List[Dict[str, Any]]) -> bool:
        mute_words = ["sqrt", "mean", "@", "com", "http", "site:", "markdown"]
        try:
            for tmp_line in test_data:
                content = tmp_line['data'][0]['content'].lower() + tmp_line['data'][1]['content'].lower()
                if any(kw in content for kw in _CODE_KEYWORDS):
                    continue

                test_assi_str = ''.join(
                    sub['content'] for sub in tmp_line['data'] if sub['role'] == 'assistant'
                )
                test_user_str = ''.join(
                    sub['content'] for sub in tmp_line['data'] if sub['role'] == 'user'
                )

                find_zh_en = re.findall(r'[\u4e00-\u9fff]([a-zA-Z]+)', test_assi_str)
                find_en_zh = re.findall(r'([a-zA-Z]+)[\u4e00-\u9fff]', test_assi_str)
                find_pattern = [p for p in find_zh_en + find_en_zh if len(p) >= 2]

                if not find_pattern or len(test_assi_str) < 150 or self.english_data(test_assi_str):
                    continue

                filtered = [p for p in find_pattern if not any(m in p.lower() for m in mute_words)]
                if not filtered:
                    continue

                count_from_user = sum(1 for p in filtered if p in test_user_str)
                if count_from_user < len(filtered) * 0.5:
                    return True
        except Exception:
            return True
        return False

    def other_gpt_keywords(self, test_data: List[Dict[str, Any]]) -> bool:
        try:
            for tmp_line in test_data:
                for idx, sub_data in enumerate(tmp_line['data']):
                    if sub_data['role'] == 'assistant':
                        assistant_content = str(sub_data['content'])
                        user_content = str(tmp_line['data'][idx - 1]['content']) if idx > 0 else ''
                        if _GPT_KEYWORDS.search(assistant_content) and not _GPT_KEYWORDS.search(user_content):
                            return True
        except Exception:
            return True
        return False

    def think_checking(self, test_data: List[Dict[str, Any]]) -> bool:
        try:
            for tmp_line in test_data:
                test_str = ''.join(
                    sub['content'] for sub in tmp_line['data'] if sub['role'] == 'assistant'
                )
                match = _THINK_PATTERN.search(test_str)
                if match is None or match.group(1).strip() == "":
                    return True
        except Exception:
            return True
        return False

    def math_answer_checking(self, test_data: List[Dict[str, Any]]) -> bool:
        try:
            for temp_data in test_data:
                answer = temp_data.get('ref_answer')
                if answer is None:
                    return False
                pangu_3000 = temp_data['data'][1].get('content', '')
                boxed_match = _BOXED_PATTERN.search(pangu_3000)
                if not boxed_match:
                    return False

                boxed_result_raw = boxed_match.group(1)
                boxed_result_clean = boxed_result_raw.replace(' ', '').replace(r'\dfrac', r'\frac')
                answer_clean = str(answer).replace(' ', '').replace(r'\dfrac', r'\frac')

                if _CHINESE_CHAR.search(answer_clean) and not re.search(r'\\frac', boxed_result_clean):
                    answer_nums = re.findall(r'\d+', answer_clean)
                    boxed_nums = re.findall(r'\d+', boxed_result_clean)
                    if answer_nums and boxed_nums and answer_nums[-1] == boxed_nums[-1]:
                        return True

                return boxed_result_clean == answer_clean
        except Exception:
            return False
        return False

    def fk_yes_or_no_checking(self, test_data: List[Dict[str, Any]]) -> bool:
        try:
            for temp_data in test_data:
                assistant_content = temp_data['data'][1].get('content', '')
                match = _ANSWER_TAG_PATTERN.search(assistant_content)
                if match:
                    result = match.group(1).strip()
                    if result in ('是', '否'):
                        return True
                tail = assistant_content.strip()[-15:].replace("是否", "")
                if ('是' in tail and '否' in tail) or '不是' in tail:
                    continue
                if '是' in tail or '否' in tail:
                    return True
        except Exception:
            return True
        return False

    def fk_answer_checking(self, test_data: List[Dict[str, Any]]) -> bool:
        try:
            for temp_data in test_data:
                assistant_content = temp_data['data'][1].get('content', '')
                ref_answer = temp_data.get('ref_answer')
                if ref_answer is None:
                    continue
                match = _ANSWER_TAG_PATTERN.search(assistant_content)
                if match:
                    model_answer = match.group(1).strip()
                else:
                    final_match = re.search(r'(?:.*\s+|^)(是|否)(?:\s+.*|$)', assistant_content.strip())
                    model_answer = final_match.group(1) if final_match else ''
                return str(model_answer) == str(ref_answer)
        except Exception:
            return False
        return False
