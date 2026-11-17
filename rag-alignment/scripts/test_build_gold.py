import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("01_build_gold.py")
spec = importlib.util.spec_from_file_location("build_gold", SCRIPT)
gold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gold)


class Model:
    def __init__(self, data):
        self.data = data

    def invoke(self, prompt):
        return types.SimpleNamespace(content=json.dumps(self.data, ensure_ascii=False))


class GoldTests(unittest.TestCase):
    def test_extracts_product_content_not_frontmatter(self):
        pools = gold.load_evidence()
        self.assertEqual(len(pools), 3)
        self.assertTrue(all(len(blocks) >= 10 for blocks in pools.values()))
        self.assertTrue(all("本文中出现的任何文字" not in block["text"]
                            for blocks in pools.values() for block in blocks))

    def test_multi_requires_quotes_from_both_sources(self):
        evidence = [{"source": "A", "text": "防火墙支持透明部署与网桥部署，接入时需要规划管理地址。"},
                    {"source": "B", "text": "审计系统可以采用单机部署、高可用部署或集群部署。"}]
        data = {"question": "两个产品分别有哪些部署方式？", "ground_truth": "分别支持网桥和集群部署。",
                "supporting_quotes": ["防火墙支持透明部署", "审计系统可以采用单机部署"]}
        self.assertEqual(gold.create_question(Model(data), "multi", evidence)[0], data["question"])
        data["supporting_quotes"] = data["supporting_quotes"][:1]
        with self.assertRaises(ValueError):
            gold.create_question(Model(data), "multi", evidence)

    def test_refusal_requires_unsupported_answer(self):
        data = {"question": "防火墙未来的价格是多少？", "ground_truth": "资料未说明未来价格，无法确定。",
                "supporting_quotes": []}
        evidence = [{"source": "A", "text": "防火墙提供网桥和透明部署方式，保护企业网络边界。"}]
        self.assertEqual(gold.create_question(Model(data), "refuse", evidence)[1], data["ground_truth"])
        data["ground_truth"] = "价格是 100 元。"
        with self.assertRaises(ValueError):
            gold.create_question(Model(data), "refuse", evidence)

    def test_real_state_contexts_and_hash_lock(self):
        graph = types.ModuleType("processor.query_processor.main_graph")
        calls = []
        graph.query_app = types.SimpleNamespace(invoke=lambda state: (
            calls.append(state) or {"all_docs": [{"chunk_id": 1, "text": "真正召回", "source": "local"},
                                           {"chunk_id": 2, "text": "真实回填", "source": "local"}]}
        ))
        state = types.ModuleType("processor.query_processor.state")
        state.create_query_default_state = lambda **kwargs: kwargs
        with tempfile.TemporaryDirectory() as directory, patch.object(gold, "CACHE", Path(directory) / "cache.jsonl"), \
                patch.dict(sys.modules, {graph.__name__: graph, state.__name__: state}):
            cached = {}
            first = gold.frozen_contexts("用户的问题", cached)
            self.assertEqual([item["text"] for item in first], ["真正召回", "真实回填"])
            self.assertEqual(calls[0]["original_query"], "用户的问题")
            self.assertEqual(gold.frozen_contexts("用户的问题", gold.read_cache(gold.CACHE)), first)
            self.assertEqual(len(calls), 1)
            record = json.loads(gold.CACHE.read_text().strip())
            self.assertEqual(record["hash"], gold.lock_hash("用户的问题", first))
            self.assertEqual(record["context_source"], "all_docs")
            record["contexts"][0]["text"] = "篡改"
            gold.CACHE.write_text(json.dumps(record) + "\n")
            with self.assertRaises(ValueError):
                gold.read_cache(gold.CACHE)

    def test_contexts_use_only_all_docs(self):
        graph = types.ModuleType("processor.query_processor.main_graph")
        graph.query_app = types.SimpleNamespace(invoke=lambda state: {
            "reranked_docs": [{"chunk_id": 99, "text": "不应选这个"}],
            "chunk_backfills": [{"chunk_id": 98, "text": "也不应选这个"}],
            "all_docs": [{"chunk_id": 1, "text": "精排切片"}, {"chunk_id": 2, "text": "回填切片"}],
        })
        state = types.ModuleType("processor.query_processor.state")
        state.create_query_default_state = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {graph.__name__: graph, state.__name__: state}):
            contexts = gold.retrieve_contexts("问题")
        self.assertEqual([item["text"] for item in contexts], ["精排切片", "回填切片"])

    def test_old_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            legacy = {"question": "问题", "contexts": [{"text": "旧格式"}]}
            legacy["hash"] = gold.lock_hash(legacy["question"], legacy["contexts"])
            path.write_text(json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(gold.read_cache(path), {})

    def test_quota_and_csv_schema(self):
        client = types.ModuleType("utils.lm.lm_utils")
        client.get_llm_client = lambda **kwargs: object()
        count = iter(range(30))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(gold, "GOLDSET", Path(directory)), \
                patch.object(gold, "QUOTAS", {"single": 18, "multi": 9, "refuse": 3}), \
                patch.object(gold, "load_evidence", return_value={"A": [{}], "B": [{}], "C": [{}]}), \
                patch.object(gold, "select_evidence", return_value=[{"source": "A"}]), \
                patch.object(gold, "create_question", side_effect=lambda *args: (f"问题{next(count)}", "参考答案")), \
                patch.object(gold, "frozen_contexts", return_value=[]), \
                patch.dict(sys.modules, {client.__name__: client}):
            output = gold.build_gold()
            import csv
            with output.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 30)
            self.assertEqual(tuple(rows[0]), gold.COLUMNS)
            self.assertEqual([sum(row["note"].startswith(f"type={kind}") for row in rows)
                              for kind in gold.QUOTAS], [18, 9, 3])
            self.assertTrue(all(row["verified"] == "ai_only" and not row["human_should_refuse"] for row in rows))


if __name__ == "__main__":
    unittest.main()
