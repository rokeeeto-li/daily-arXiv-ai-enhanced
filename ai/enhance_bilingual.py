"""
双语增强版本 - 生成中英文双语摘要
基于原版 enhance.py 修改
"""
import os
import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import requests

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()

def process_single_item_bilingual(chain_zh, chain_en, item: Dict) -> Dict:
    """处理单个数据项 - 生成双语摘要"""
    def is_sensitive(content: str) -> bool:
        try:
            resp = requests.post(
                "https://spam.dw-dengwei.workers.dev",
                json={"text": content},
                timeout=5
            )
            if resp.status_code == 200:
                result = resp.json()
                return result.get("sensitive", True)
            else:
                print(f"Sensitive check failed with status {resp.status_code}", file=sys.stderr)
                return True
        except Exception as e:
            print(f"Sensitive check error: {e}", file=sys.stderr)
            return True

    def check_github_code(content: str) -> Dict:
        code_info = {}
        github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
        match = re.search(github_pattern, content)

        if match:
            owner, repo = match.groups()
            repo = repo.rstrip(".git").rstrip(".,)")
            full_url = f"https://github.com/{owner}/{repo}"
            code_info["code_url"] = full_url

            github_token = os.environ.get("TOKEN_GITHUB")
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"

            try:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                resp = requests.get(api_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    code_info["code_stars"] = data.get("stargazers_count", 0)
                    code_info["code_last_update"] = data.get("pushed_at", "")[:10]
            except Exception:
                pass
            return code_info

        github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
        match_io = re.search(github_io_pattern, content)

        if match_io:
            url = match_io.group(0)
            url = url.rstrip(".,)")
            code_info["code_url"] = url

        return code_info

    # 检查敏感内容
    if is_sensitive(item.get("summary", "")):
        return None

    # 检测 GitHub 代码
    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    # Default fallback
    default_ai_fields = {
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed"
    }

    def invoke_chain_safe(chain, lang):
        """安全调用链"""
        try:
            response: Structure = chain.invoke({
                "language": lang,
                "content": item['summary']
            })
            return response.model_dump()
        except langchain_core.exceptions.OutputParserException as e:
            error_msg = str(e)
            partial_data = {}

            if "Function Structure arguments:" in error_msg:
                try:
                    json_str = error_msg.split("Function Structure arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
                    json_str = json_str.replace('\\', '\\\\')
                    partial_data = json.loads(json_str)
                except Exception as json_e:
                    print(f"Failed to parse JSON for {item.get('id', 'unknown')}: {json_e}", file=sys.stderr)

            return {**default_ai_fields, **partial_data}
        except Exception as e:
            print(f"Unexpected error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
            return default_ai_fields

    # 生成中文摘要
    ai_zh = invoke_chain_safe(chain_zh, "Chinese")

    # 生成英文摘要
    ai_en = invoke_chain_safe(chain_en, "English")

    # 合并为双语格式
    item['AI'] = {
        'tldr_zh': ai_zh.get('tldr', ''),
        'tldr_en': ai_en.get('tldr', ''),
        'motivation_zh': ai_zh.get('motivation', ''),
        'motivation_en': ai_en.get('motivation', ''),
        'method_zh': ai_zh.get('method', ''),
        'method_en': ai_en.get('method', ''),
        'result_zh': ai_zh.get('result', ''),
        'result_en': ai_en.get('result', ''),
        'conclusion_zh': ai_zh.get('conclusion', ''),
        'conclusion_en': ai_en.get('conclusion', '')
    }

    # 检查敏感内容
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return None

    return item

def process_all_items_bilingual(data: List[Dict], model_name: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项 - 双语版本"""
    llm_zh = ChatOpenAI(model=model_name).with_structured_output(Structure, method="function_calling")
    llm_en = ChatOpenAI(model=model_name).with_structured_output(Structure, method="function_calling")

    print(f'Connect to: {model_name} (Bilingual mode)', file=sys.stderr)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain_zh = prompt_template | llm_zh
    chain_en = prompt_template | llm_en

    # 使用线程池并行处理
    processed_data = [None] * len(data)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_item_bilingual, chain_zh, chain_en, item): idx
            for idx, item in enumerate(data)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items (bilingual)"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                processed_data[idx] = data[idx]
                processed_data[idx]['AI'] = {
                    "tldr_zh": "处理失败", "tldr_en": "Processing failed",
                    "motivation_zh": "处理失败", "motivation_en": "Processing failed",
                    "method_zh": "处理失败", "method_en": "Processing failed",
                    "result_zh": "处理失败", "result_en": "Processing failed",
                    "conclusion_zh": "处理失败", "conclusion_en": "Processing failed"
                }

    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'GLM-4.7-Flash')

    # 生成双语文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_Bilingual.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)

    # 并行处理所有数据（双语）
    processed_data = process_all_items_bilingual(
        data,
        model_name,
        args.max_workers
    )

    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

    print(f'\n✅ Bilingual summaries generated: {target_file}', file=sys.stderr)

if __name__ == "__main__":
    main()
