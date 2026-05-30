"""加载 LLM 参数优化所需的数据集分析报告等上下文。"""
import os


def load_dataset_report(project_root: str) -> str:
    report_path = os.path.join(
        project_root, 'data', 'results', 'dataset_analysis_report.json'
    )
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            '未找到 dataset_analysis_report.json，请先在「数据集管理」页上传并分析数据'
        )
    with open(report_path, 'r', encoding='utf-8') as f:
        return f.read()
