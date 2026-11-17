"""使用 Ragas 评估项目 RAG 查询流程。

默认读取预设的 QA 数据集，分批查询、逐题保存查询检查点并逐题评分；同一输出路径可自动续跑。
评估时直接调用项目查询图，从 node_answer_output 的 all_docs 读取上下文与最终回答，
不修改原有业务代码。

运行方式：
    python eval/script/eval.py
    python eval/script/eval.py --output eval/data/other.csv --batch-size 2
    python eval/script/eval.py --limit 2

批大小、结果文件和检查点路径可在 eval/.env 配置；中断后用相同配置重启。

配置 EVAL_QUERY_MODEL_PATH 时仅替换最终答案模型；意图识别、HyDE 和裁判保持原配置。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import math
import os
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values
from openai import OpenAI

from common.config.lm_config import lm_config
from utils.lm.embedding_utils import generate_embeddings


LOGGER = logging.getLogger("ragas_eval")

DEFAULT_INPUT_CSV = PROJECT_ROOT / "eval" / 'rag-alignment'/'data'/'goldset'/ 'Holdout-1116-1.csv'
EVAL_ENV_PATH = PROJECT_ROOT / "eval" / ".env"
JUDGE_CONFIG = dotenv_values(EVAL_ENV_PATH, interpolate=False)


def config_path(name: str, default: str) -> Path:
    path = Path(JUDGE_CONFIG.get(name) or default).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


DEFAULT_BATCH_SIZE = int(JUDGE_CONFIG.get("EVAL_BATCH_SIZE") or "5")
DEFAULT_OUTPUT_CSV = config_path("EVAL_OUTPUT_CSV", "eval/data/eval-checkpoint/Holdout-1114-qwen3-4b-train-before.csv")
DEFAULT_CHECKPOINT_PATH = config_path(
    "EVAL_CHECKPOINT_PATH", str(DEFAULT_OUTPUT_CSV) + ".checkpoint.json"
)
EVAL_JUDGE_MODEL = JUDGE_CONFIG.get("EVAL_JUDGE_MODEL")
EVAL_JUDGE_BASE_URL = JUDGE_CONFIG.get("EVAL_JUDGE_BASE_URL")
EVAL_JUDGE_API_KEY = JUDGE_CONFIG.get("EVAL_JUDGE_API_KEY")
EVAL_JUDGE_TEMPERATURE = float(JUDGE_CONFIG.get("EVAL_JUDGE_TEMPERATURE") or "0")
EVAL_JUDGE_TOP_P = float(JUDGE_CONFIG.get("EVAL_JUDGE_TOP_P") or "1")
EVAL_QUERY_MODEL_PATH = JUDGE_CONFIG.get("EVAL_QUERY_MODEL_PATH")
EVAL_QUERY_BASE_URL = JUDGE_CONFIG.get("EVAL_QUERY_BASE_URL") or "https://pytrio.com/api/openai/v1"
EVAL_QUERY_API_KEY = JUDGE_CONFIG.get("EVAL_QUERY_API_KEY") or os.getenv("TRIO_API_KEY")

OUTPUT_COLUMNS = [
    "question",
    "context",
    "Answer",
    "ground_truth",
    "faithfulness",
    "Answer Relevancy",
    "Context Precision",
    "Context Recall",
    "Answer Correctness",
    "status",
    "error",
]
SCORE_COLUMNS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "answer_correctness": "Answer Correctness",
}


def validate_model_configs() -> None:
    if not EVAL_ENV_PATH.is_file():
        raise FileNotFoundError(f"裁判模型配置文件不存在：{EVAL_ENV_PATH}")
    missing = [key for key in ("EVAL_JUDGE_MODEL", "EVAL_JUDGE_BASE_URL", "EVAL_JUDGE_API_KEY") if not JUDGE_CONFIG.get(key)]
    if missing:
        raise ValueError(f"请在 eval/.env 中填写裁判模型配置：{', '.join(missing)}")
    if (EVAL_JUDGE_MODEL, EVAL_JUDGE_BASE_URL) == (lm_config.llm_model, lm_config.base_url):
        raise ValueError("裁判模型与查询图模型相同：请在 eval/.env 中配置不同的模型或服务地址")
    if EVAL_QUERY_MODEL_PATH and not EVAL_QUERY_API_KEY:
        raise ValueError("已配置 EVAL_QUERY_MODEL_PATH；请配置 EVAL_QUERY_API_KEY 或环境变量 TRIO_API_KEY")


def run_query_pipeline_with_configured_model(
    qa_rows: Sequence[dict[str, str]], on_row_completed=None
) -> list[dict[str, Any]]:
    if not EVAL_QUERY_MODEL_PATH:
        return step_2_run_query_pipeline(qa_rows, on_row_completed=on_row_completed)

    answer_module = importlib.import_module(
        "processor.query_processor.nodes.node_answer_output"
    )
    original_getter = answer_module.get_llm_client
    clients = {}

    class PyTRIOAnswerClient:
        def __init__(self, temperature: float, top_p: float):
            self.client = OpenAI(base_url=EVAL_QUERY_BASE_URL, api_key=EVAL_QUERY_API_KEY)
            self.temperature = temperature
            self.top_p = top_p

        @staticmethod
        def content_or_reasoning(message) -> str:
            return message.content or getattr(message, "reasoning", None) or ""

        def invoke(self, prompt: str) -> AIMessage:
            response = self.client.chat.completions.create(
                model=EVAL_QUERY_MODEL_PATH,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
            )
            content = self.content_or_reasoning(response.choices[0].message)
            if not content.strip():
                raise RuntimeError("PyTRIO 推理权重返回空回答（content/reasoning 均为空）")
            usage = response.usage.model_dump() if response.usage else {}
            return AIMessage(content=content, response_metadata={"token_usage": usage})

        def stream(self, prompt: str):
            response = self.client.chat.completions.create(
                model=EVAL_QUERY_MODEL_PATH,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True,
            )
            for choice in response:
                if choice.choices:
                    content = self.content_or_reasoning(choice.choices[0].delta)
                    if content:
                        yield AIMessageChunk(content=content)

    def get_eval_query_client(model=None, json_mode=False, temperature=None, top_p=None):
        if model is not None and model != lm_config.llm_model:
            return original_getter(model=model, json_mode=json_mode, temperature=temperature, top_p=top_p)
        effective_temperature = lm_config.llm_temperature if temperature is None else temperature
        effective_top_p = lm_config.llm_top_p if top_p is None else top_p
        cache_key = (json_mode, effective_temperature, effective_top_p)
        if cache_key not in clients:
            clients[cache_key] = PyTRIOAnswerClient(effective_temperature, effective_top_p)
        return clients[cache_key]

    try:
        answer_module.get_llm_client = get_eval_query_client
        LOGGER.info(
            "仅最终答案使用 PyTRIO 权重：%s；意图识别/HyDE 保持原模型：%s",
            EVAL_QUERY_MODEL_PATH, lm_config.llm_model,
        )
        return step_2_run_query_pipeline(qa_rows, on_row_completed=on_row_completed)
    finally:
        answer_module.get_llm_client = original_getter


class ProjectEmbeddingAdapter(Embeddings):
    """将项目的混合向量工具适配为 Ragas 所需的 LangChain Embeddings 接口。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """调用项目 embedding 工具批量生成稠密向量。"""
        if not texts:
            return []
        safe_texts = [text if text and text.strip() else " " for text in texts]
        return generate_embeddings(safe_texts)["dense"]

    def embed_query(self, text: str) -> list[float]:
        """调用项目 embedding 工具生成单条查询的稠密向量。"""
        return self.embed_documents([text])[0]


def parse_args() -> argparse.Namespace:
    """解析输入输出路径及调试、并发参数。"""
    parser = argparse.ArgumentParser(description="运行项目 RAG 的 Ragas 评估")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="输入 QA CSV，需包含 question、ground_truth")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="评估结果 CSV 路径")
    parser.add_argument("--limit", type=int, default=None, help="仅运行前 N 条，用于小样本验证")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="每批查询和 Ragas 评估的问题数")
    parser.add_argument("--checkpoint", type=Path, default=None, help="检查点 JSON 路径，默认跟随输出路径")
    parser.add_argument("--timeout", type=int, default=300, help="单项 Ragas 任务超时秒数")
    parser.add_argument("--max-workers", type=int, default=1, help="Ragas 评估最大并发数")
    parser.add_argument("--metric-retries", type=int, default=2, help="每个缺失指标的额外重试次数")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = (
            DEFAULT_CHECKPOINT_PATH if args.output.resolve() == DEFAULT_OUTPUT_CSV.resolve()
            else args.output.with_suffix(args.output.suffix + ".checkpoint.json")
        )
    return args


def step_1_load_qa_csv(input_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """读取 QA CSV，校验必需列和每行的问题、标准答案。"""
    if not input_path.is_file():
        raise FileNotFoundError(f"评估输入文件不存在：{input_path}")

    qa_rows: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"question", "ground_truth"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"输入 CSV 缺少列：{', '.join(sorted(missing_columns))}")

        for row_number, row in enumerate(reader, start=2):
            question = (row.get("question") or "").strip()
            ground_truth = (row.get("ground_truth") or "").strip()
            if not question or not ground_truth:
                raise ValueError(f"第 {row_number} 行 question 或 ground_truth 为空")
            qa_rows.append({"question": question, "ground_truth": ground_truth})
            if limit is not None and len(qa_rows) >= limit:
                break

    if not qa_rows:
        raise ValueError(f"没有从 {input_path} 读取到有效 QA 数据")
    LOGGER.info("读取到 %s 条评估问题", len(qa_rows))
    return qa_rows


def step_2_run_query_pipeline(
    qa_rows: Sequence[dict[str, str]],
    on_row_completed=None,
) -> list[dict[str, Any]]:
    """逐题运行查询图；单题失败记入结果，成功一题后立即调用持久化回调。"""
    from processor.query_processor.main_graph import query_app
    from processor.query_processor.state import create_query_default_state

    evaluated_rows: list[dict[str, Any]] = []

    for index, row in enumerate(qa_rows, start=1):
        question = row["question"]
        session_id = f"ragas_eval_{uuid.uuid4().hex}"
        all_docs: list[Any] = []
        answer = ""
        error = ""

        LOGGER.info("执行 RAG 查询 [%s/%s]：%s", index, len(qa_rows), question)
        try:
            state = create_query_default_state(
                session_id=session_id,
                original_query=question,
                is_stream=False,
                query_start_ts=time.time(),
            )
            for node_update in query_app.stream(state, stream_mode="updates"):
                answer_update = node_update.get("node_answer_output")
                if answer_update is not None:
                    all_docs = answer_update.get("all_docs") or []
                    answer = str(answer_update.get("answer") or "")
            if not answer.strip():
                raise RuntimeError("查询图未返回答案")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("查询失败，继续处理后续问题：%s", question)

        contexts = step_2_extract_contexts(all_docs)
        evaluated_row = {
            "question": question,
            "ground_truth": row["ground_truth"],
            "answer": answer,
            "contexts": contexts,
            "context": step_2_format_contexts(contexts),
            "status": "query_failed" if error else "query_succeeded",
            "error": error,
        }
        evaluated_rows.append(evaluated_row)
        if on_row_completed:
            on_row_completed(evaluated_row)
        LOGGER.info(
            "查询%s：答案长度=%s，最终上下文数=%s",
            "失败" if error else "完成",
            len(answer),
            len(contexts),
        )

    return evaluated_rows


def step_2_extract_contexts(all_docs: Any) -> list[str]:
    """从 node_answer_output 的 all_docs 中提取 Ragas 所需的文本上下文列表。"""
    if not all_docs:
        return []
    if isinstance(all_docs, dict):
        all_docs = all_docs.get("docs") or all_docs.get("contexts") or []
    if isinstance(all_docs, str):
        return [all_docs.strip()] if all_docs.strip() else []

    contexts: list[str] = []
    for document in all_docs:
        if isinstance(document, str):
            content = document
        elif isinstance(document, dict):
            content = (
                document.get("text")
                or document.get("content")
                or document.get("page_content")
                or document.get("snippet")
                or ""
            )
        else:
            content = getattr(document, "page_content", "")
        content = str(content).strip()
        if content:
            contexts.append(content)
    return contexts


def step_2_format_contexts(contexts: Sequence[str]) -> str:
    """将上下文列表转换为便于人工查看且保留分段的 CSV 字段。"""
    return "\n\n".join(
        f"[{index}] {context}" for index, context in enumerate(contexts, start=1)
    )


def step_3_build_ragas_dataset(evaluated_rows: Sequence[dict[str, Any]]) -> Any:
    """把查询结果映射成 Ragas 单轮评估数据集字段。"""
    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI
    except ModuleNotFoundError:
        vertex_module = types.ModuleType("langchain_community.chat_models.vertexai")
        vertex_module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[vertex_module.__name__] = vertex_module
    from ragas import EvaluationDataset

    samples = [
        {
            "user_input": row["question"],
            "retrieved_contexts": row["contexts"],
            "response": row["answer"],
            "reference": row["ground_truth"],
        }
        for row in evaluated_rows
    ]
    return EvaluationDataset.from_list(samples)


def step_4_run_ragas_evaluation(
    dataset: Any,
    batch_size: int,
    timeout: int,
    max_workers: int,
    metric_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """使用 eval/.env 中的裁判模型、项目 embedding 和五项 Ragas 指标执行评估。"""
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        answer_correctness,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    llm = ChatOpenAI(
        model=EVAL_JUDGE_MODEL,
        base_url=EVAL_JUDGE_BASE_URL,
        api_key=EVAL_JUDGE_API_KEY,
        temperature=EVAL_JUDGE_TEMPERATURE,
        top_p=EVAL_JUDGE_TOP_P,
    )
    embeddings = ProjectEmbeddingAdapter()
    all_metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": AnswerRelevancy(strictness=1),
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }
    unknown_metrics = (metric_names or set()) - set(all_metrics)
    if unknown_metrics:
        raise ValueError(f"未知 Ragas 指标：{', '.join(sorted(unknown_metrics))}")
    metrics = [metric for name, metric in all_metrics.items()
               if metric_names is None or name in metric_names]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=timeout, max_workers=max_workers, max_retries=3),
        raise_exceptions=False,
        show_progress=True,
        batch_size=batch_size,
    )
    return result.scores


def step_5_normalize_score(value: Any) -> float | str:
    """将 Ragas 分数规范为 CSV 可写的数值或空值。"""
    if value is None:
        return ""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return ""
    return round(score, 6) if math.isfinite(score) else ""


def checkpoint_path_for(output_path: Path, configured_path: Path | None = None) -> Path:
    return configured_path or output_path.with_suffix(output_path.suffix + ".checkpoint.json")


def save_checkpoint(checkpoint_path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as checkpoint_file:
        json.dump(checkpoint, checkpoint_file, ensure_ascii=False)
        checkpoint_file.flush()
        os.fsync(checkpoint_file.fileno())
    os.replace(temporary_path, checkpoint_path)


def save_completed_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        csv_file.flush()
        os.fsync(csv_file.fileno())
    os.replace(temporary_path, output_path)


def score_batch(rows: Sequence[dict[str, Any]], scores: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(scores) != len(rows):
        raise ValueError(f"Ragas 返回 {len(scores)} 条评分，预期 {len(rows)} 条；本批未提交")
    output_rows = []
    for row, score in zip(rows, scores):
        if not isinstance(score, dict):
            score = {}
        output_row = {
            "question": row["question"],
            "context": row["context"],
            "Answer": row["answer"],
            "ground_truth": row["ground_truth"],
            "status": "completed",
            "error": "",
        }
        output_row.update({
            column: step_5_normalize_score(score.get(metric))
            for metric, column in SCORE_COLUMNS.items()
        })
        output_rows.append(output_row)
    return output_rows


def evaluate_row_with_retries(
    row: dict[str, Any], timeout: int, max_workers: int, metric_retries: int,
) -> dict[str, Any]:
    """先评全部适用指标，再单独重试 NaN/缺失指标，保留已得到的分数。"""
    context_metrics = {"faithfulness", "context_precision", "context_recall"}
    metric_names = set(SCORE_COLUMNS)
    skipped = set()
    if not row.get("contexts"):
        skipped = metric_names & context_metrics
        metric_names -= skipped

    collected: dict[str, Any] = {}
    failures: dict[str, str] = {}
    pending = metric_names
    for attempt in range(metric_retries + 1):
        if not pending:
            break
        try:
            scores = step_4_run_ragas_evaluation(
                step_3_build_ragas_dataset([row]),
                batch_size=1,
                timeout=timeout,
                max_workers=max_workers,
                metric_names=pending,
            )
            if len(scores) != 1:
                raise ValueError(f"Ragas 返回 {len(scores)} 条评分，预期 1 条")
            score = scores[0] if isinstance(scores[0], dict) else {}
            for metric in pending:
                normalized = step_5_normalize_score(score.get(metric))
                if normalized != "":
                    collected[metric] = normalized
                    failures.pop(metric, None)
                else:
                    failures[metric] = "Ragas 返回空值或 NaN"
        except Exception as exc:
            for metric in pending:
                failures[metric] = f"{type(exc).__name__}: {exc}"
        pending = {metric for metric in pending if metric not in collected}
        if pending and attempt < metric_retries:
            LOGGER.warning("以下指标缺失，将进行第 %s/%s 次单项重试：%s",
                           attempt + 1, metric_retries, ", ".join(sorted(pending)))

    output_row = score_batch([row], [collected])[0]
    missing = [SCORE_COLUMNS[metric] for metric in metric_names if metric not in collected]
    output_row["status"] = "evaluation_partial" if missing or skipped else "completed"
    details = [f"{SCORE_COLUMNS[metric]}: {failures.get(metric, '重试后仍缺失')}"
               for metric in missing]
    if skipped:
        details.append("无检索上下文，未计算：" + ", ".join(
            SCORE_COLUMNS[metric] for metric in sorted(skipped)
        ))
    output_row["error"] = "; ".join(details)
    return output_row


def main() -> None:
    """按批查询、评分、持久化并从检查点续跑。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size 必须大于 0")
    if args.metric_retries < 0:
        raise ValueError("--metric-retries 不能小于 0")
    if args.output.resolve() == args.checkpoint.resolve():
        raise ValueError("结果 CSV 与检查点不能使用同一个路径")
    validate_model_configs()
    qa_rows = step_1_load_qa_csv(args.input, limit=args.limit)
    LOGGER.info(
        "查询图模型：%s；裁判模型：%s",
        EVAL_QUERY_MODEL_PATH or lm_config.llm_model,
        EVAL_JUDGE_MODEL,
    )
    fingerprint_config = {
        "input": qa_rows,
        "query_model": EVAL_QUERY_MODEL_PATH or lm_config.llm_model,
        "query_base_url": EVAL_QUERY_BASE_URL if EVAL_QUERY_MODEL_PATH else lm_config.base_url,
        "judge_model": EVAL_JUDGE_MODEL,
        "judge_url": EVAL_JUDGE_BASE_URL,
    }
    legacy_fingerprints = {
        hashlib.sha256(json.dumps(
            fingerprint_config, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")).hexdigest()
    }
    if EVAL_QUERY_MODEL_PATH:
        fingerprint_config.update({
            "query_model_scope": "answer_only_v2",
            "retrieval_model": lm_config.llm_model,
            "retrieval_base_url": lm_config.base_url,
        })
        previous_config = dict(fingerprint_config, query_model_scope="answer_only_v1")
        legacy_fingerprints.add(hashlib.sha256(json.dumps(
            previous_config, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")).hexdigest())
    fingerprint = hashlib.sha256(json.dumps(
        fingerprint_config, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")).hexdigest()
    checkpoint_path = checkpoint_path_for(args.output, args.checkpoint)
    if EVAL_QUERY_MODEL_PATH and checkpoint_path.exists():
        old_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if old_checkpoint.get("fingerprint") in legacy_fingerprints:
            suffix = f"-answer-{fingerprint[:10]}"
            original_path = checkpoint_path
            args.output = args.output.with_name(args.output.stem + suffix + args.output.suffix)
            checkpoint_path = checkpoint_path.with_name(
                checkpoint_path.stem + suffix + checkpoint_path.suffix
            )
            LOGGER.warning(
                "保留旧版检查点 %s；修正后的答案评估使用 %s，检查点 %s",
                original_path, args.output, checkpoint_path,
            )
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError(f"检查点与当前输入或模型不匹配，请换 --output：{checkpoint_path}")
    else:
        if args.output.exists():
            raise FileExistsError(f"结果文件已存在但无检查点，拒绝覆盖：{args.output}")
        checkpoint = {"fingerprint": fingerprint, "completed": [], "pending": []}
        save_checkpoint(checkpoint_path, checkpoint)

    completed = checkpoint["completed"]
    if len(completed) > len(qa_rows) or any(
        (row["question"], row["ground_truth"]) != (qa_rows[index]["question"], qa_rows[index]["ground_truth"])
        for index, row in enumerate(completed)
    ):
        raise ValueError("检查点已完成记录与输入不一致")
    pending = checkpoint["pending"]
    if len(completed) + len(pending) > len(qa_rows) or any(
        (row["question"], row["ground_truth"])
        != (qa_rows[len(completed) + index]["question"], qa_rows[len(completed) + index]["ground_truth"])
        for index, row in enumerate(pending)
    ):
        raise ValueError("检查点待评分记录与输入不一致")
    save_completed_csv(completed, args.output)
    while len(completed) < len(qa_rows):
        pending = checkpoint["pending"]
        if not pending:
            batch_rows = qa_rows[len(completed):len(completed) + args.batch_size]

            def persist_query_row(row):
                # 回调置于查询异常捕获之外：磁盘写入失败必须终止，不能当作模型失败吞掉。
                pending.append(row)
                save_checkpoint(checkpoint_path, checkpoint)

            run_query_pipeline_with_configured_model(
                batch_rows, on_row_completed=persist_query_row
            )

        # pending 始终是输入的连续前缀；恢复时先处理已落盘题，再发起新查询。
        while pending:
            row = pending[0]
            error = row.get("error", "")
            status = row.get("status", "query_succeeded")
            if status != "query_failed":
                try:
                    output_row = evaluate_row_with_retries(
                        row, timeout=args.timeout, max_workers=args.max_workers,
                        metric_retries=args.metric_retries,
                    )
                except Exception as exc:
                    status = "evaluation_failed"
                    error = f"{type(exc).__name__}: {exc}"
                    LOGGER.exception("评分失败，保留答案并继续：%s", row["question"])
            if status in ("query_failed", "evaluation_failed"):
                output_row = {
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "context": row["context"],
                    "Answer": row["answer"],
                    **{column: "" for column in SCORE_COLUMNS.values()},
                    "status": status,
                    "error": error,
                }
            completed.append(output_row)
            pending.pop(0)
            save_checkpoint(checkpoint_path, checkpoint)
            save_completed_csv(completed, args.output)
            LOGGER.info("已保存 %s/%s 条：%s", len(completed), len(qa_rows), args.output)

    failed_count = sum(row.get("status") in ("query_failed", "evaluation_failed") for row in completed)
    partial_count = sum(row.get("status") == "evaluation_partial" for row in completed)
    LOGGER.info("评测结束：共 %s 题，失败 %s 题，部分指标缺失 %s 题", len(completed), failed_count, partial_count)


if __name__ == "__main__":
    main()
