"""从实际 RAG 回答调用采集 Prompt，生成 ShareGPT JSONL 和回填 CSV。

默认不覆盖源 CSV；运行前需启动查询图依赖的模型、向量库和会话数据库。
"""

import argparse
import csv
import importlib
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NO_SUBJECT_ANSWER = "没有可信和可选的主体，请明确以后再次提问!!"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DATA_DIR / "train/train-1116.csv")
    parser.add_argument("--val-csv", type=Path, default=DATA_DIR / "val/val-1116.csv")
    parser.add_argument("--train-jsonl", type=Path, help="默认与训练 CSV 同目录、同名 .jsonl")
    parser.add_argument("--val-jsonl", type=Path, help="默认与验证 CSV 同目录、同名 .jsonl")
    parser.add_argument("--train-output-csv", type=Path, help="默认生成 *-with-input.csv")
    parser.add_argument("--val-output-csv", type=Path, help="默认生成 *-with-input.csv")
    parser.add_argument("--overwrite-source", action="store_true", help="原位回填两份源 CSV")
    parser.add_argument("--query-column", default="query")
    parser.add_argument("--answer-column", default="target/ground_truth")
    parser.add_argument("--input-column", default="input")
    parser.add_argument("--messages-key", default="messages")
    parser.add_argument("--role-key", default="role")
    parser.add_argument("--content-key", default="content")
    parser.add_argument("--user-role", default="user")
    parser.add_argument("--assistant-role", default="assistant")
    parser.add_argument("--session-prefix", default="sft_data")
    parser.add_argument("--limit", type=int,help="仅处理每份 CSV 的前 N 行，用于链路试运行")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须是正整数")
    if args.overwrite_source and (args.train_output_csv or args.val_output_csv):
        parser.error("--overwrite-source 不能与 --train-output-csv / --val-output-csv 同时使用")
    for split in ("train", "val"):
        source = getattr(args, f"{split}_csv")
        setattr(args, f"{split}_jsonl", getattr(args, f"{split}_jsonl") or source.with_suffix(".jsonl"))
        output_csv = source if args.overwrite_source else source.with_name(f"{source.stem}-with-input.csv")
        setattr(args, f"{split}_output_csv", getattr(args, f"{split}_output_csv") or output_csv)

    inputs = [args.train_csv.resolve(), args.val_csv.resolve()]
    outputs = [args.train_jsonl.resolve(), args.val_jsonl.resolve(),
               args.train_output_csv.resolve(), args.val_output_csv.resolve()]
    if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
        parser.error("训练/验证路径及各输出路径不得重复")
    for index, output in enumerate(outputs):
        if output in inputs and not (args.overwrite_source and index >= 2
                                     and output == inputs[index - 2]):
            parser.error(f"输出文件不能覆盖输入 CSV：{output}")
    if len({args.query_column, args.answer_column, args.input_column}) != 3:
        parser.error("query、answer、input 列名必须互不相同")
    if len({args.role_key, args.content_key}) != 2:
        parser.error("role 和 content 的键名必须不同")
    return args


def read_rows(source, args):
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise ValueError(f"{source}: 缺失表头或表头重复")
        missing = {args.query_column, args.answer_column} - set(headers)
        if missing:
            raise ValueError(f"{source}: 缺少列 {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{source}: 没有数据行")
    for line_number, row in enumerate(rows, start=2):
        if None in row or not (row[args.query_column] or "").strip() or not (row[args.answer_column] or "").strip():
            raise ValueError(f"{source}:{line_number}: 行格式错误、query 或标准答案为空")
    if args.input_column not in headers:
        headers.append(args.input_column)
    return headers, rows


def invoke_and_capture(query, session_prefix, query_app, create_query_default_state,
                       answer_node, item_node):
    """捕获已渲染的回答 Prompt，不生成回答，也不写入聊天历史。"""
    prompts = []

    def capture_prompt(prompt, is_stream, session_id):
        prompts.append(prompt)
        return "", {}

    state = create_query_default_state(
        session_id=f"{session_prefix}_{uuid.uuid4().hex}",
        original_query=query,
        is_stream=False,
        query_start_ts=time.time(),
    )
    with (patch.object(answer_node, "step_5_call_llm_create_answer", capture_prompt),
          patch.object(answer_node, "step_7_save_assistant_message", lambda state: None),
          patch.object(item_node, "step_7_save_user_chat_message", lambda state: None)):
        result = query_app.invoke(state)
    if (isinstance(result, dict) and result.get("item_names") == []
            and not result.get("all_docs")):
        return query, NO_SUBJECT_ANSWER
    if not isinstance(result, dict) or not result.get("all_docs"):
        raise ValueError("RAG 未返回有效 all_docs（可能提前拒答或召回为空）")
    if len(prompts) != 1 or not isinstance(prompts[0], str) or not prompts[0].strip():
        raise ValueError("回答节点未调用模型一次，无法采集真实的完整 Prompt")
    return prompts[0], None


def write_outputs(source, jsonl_path, csv_path, args, query_app, create_query_default_state,
                  answer_node, item_node):
    headers, rows = read_rows(source, args)
    if args.limit is not None:
        rows = rows[:args.limit]
    records = []
    for index, row in enumerate(rows, start=2):
        print(f"{source.name}: {index - 1}/{len(rows)}", file=sys.stderr, flush=True)
        try:
            prompt, fallback_answer = invoke_and_capture(
                row[args.query_column], args.session_prefix,
                query_app, create_query_default_state, answer_node, item_node)
        except Exception as exc:
            raise RuntimeError(f"{source}:{index}: RAG 采集失败: {exc}") from exc
        row[args.input_column] = prompt
        if fallback_answer is not None:
            row[args.answer_column] = fallback_answer
        records.append({args.messages_key: [
            {args.role_key: args.user_role, args.content_key: prompt},
            {args.role_key: args.assistant_role, args.content_key: row[args.answer_column]},
        ]})

    temporary_files = []
    try:
        for destination in (jsonl_path, csv_path):
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                             dir=destination.parent, delete=False) as handle:
                temporary_files.append(Path(handle.name))
                if destination == jsonl_path:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    writer = csv.DictWriter(handle, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
        for temporary, destination in zip(temporary_files, (jsonl_path, csv_path)):
            os.replace(temporary, destination)
    finally:
        for temporary in temporary_files:
            temporary.unlink(missing_ok=True)
    return len(rows)


def prioritize_project_imports():
    project_root = str(PROJECT_ROOT)
    sys.path[:] = [project_root, *(entry for entry in sys.path if entry != project_root)]


def main(argv=None):
    args = parse_args(argv)
    prioritize_project_imports()
    from processor.query_processor.main_graph import query_app
    from processor.query_processor.state import create_query_default_state

    answer_node = importlib.import_module("processor.query_processor.nodes.node_answer_output")
    item_node = importlib.import_module("processor.query_processor.nodes.node_item_name_confirm")
    for split in ("train", "val"):
        source = getattr(args, f"{split}_csv")
        count = write_outputs(source, getattr(args, f"{split}_jsonl"),
                              getattr(args, f"{split}_output_csv"), args,
                              query_app, create_query_default_state, answer_node, item_node)
        print(f"{split}: {count} 条 -> {getattr(args, f'{split}_jsonl')} / {getattr(args, f'{split}_output_csv')}")


if __name__ == "__main__":
    main()
