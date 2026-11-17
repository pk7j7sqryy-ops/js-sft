import csv
import contextlib
import io
import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SOURCE = Path(__file__).with_name("sft-data.py")
SPEC = importlib.util.spec_from_file_location("sft_data", SOURCE)
sft_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sft_data)


class FakeApp:
    def __init__(self, answer_node, item_node, docs=None):
        self.answer_node = answer_node
        self.item_node = item_node
        self.docs = [{"text": "检索内容"}] if docs is None else docs
        self.sessions = []

    def invoke(self, state):
        self.sessions.append(state["session_id"])
        self.item_node.step_7_save_user_chat_message(state)
        if self.docs:
            self.answer_node.step_5_call_llm_create_answer(
                f"规则\n参考资料：{self.docs[0]['text']}\n问题：{state['original_query']}",
                state["is_stream"], state["session_id"]
            )
        self.answer_node.step_7_save_assistant_message(state)
        return {"all_docs": self.docs}


class SftDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.train = self.folder / "train.csv"
        self.val = self.folder / "val.csv"
        self.headers = ["sample_id", "query", "input", "target/ground_truth", "note"]
        for path in (self.train, self.val):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.headers)
                writer.writeheader()
                writer.writerow({"sample_id": "1", "query": "问题一", "input": "",
                                 "target/ground_truth": "标准答案", "note": "保留,此列"})

    def args(self, *extra):
        return sft_data.parse_args(["--train-csv", str(self.train), "--val-csv", str(self.val), *extra])

    def test_prompt_is_captured_from_actual_model_call_and_source_is_untouched(self):
        answer_call = Mock(side_effect=AssertionError("不应生成回答"))
        answer_save = Mock(side_effect=AssertionError("不应保存助手消息"))
        user_save = Mock(side_effect=AssertionError("不应保存用户消息"))
        answer_node = SimpleNamespace(step_5_call_llm_create_answer=answer_call,
                                      step_7_save_assistant_message=answer_save)
        item_node = SimpleNamespace(step_7_save_user_chat_message=user_save)
        original_call = answer_node.step_5_call_llm_create_answer
        app = FakeApp(answer_node, item_node)
        args = self.args()
        original = self.train.read_bytes()
        count = sft_data.write_outputs(self.train, args.train_jsonl, args.train_output_csv,
                                       args, app, lambda **kwargs: kwargs, answer_node, item_node)
        self.assertEqual(count, 1)
        self.assertEqual(self.train.read_bytes(), original)
        with args.train_jsonl.open(encoding="utf-8") as handle:
            record = json.loads(handle.readline())
            self.assertEqual(handle.readline(), "")
        prompt = "规则\n参考资料：检索内容\n问题：问题一"
        self.assertEqual(record, {"messages": [{"role": "user", "content": prompt},
                                               {"role": "assistant", "content": "标准答案"}]})
        with args.train_output_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["input"], prompt)
        self.assertEqual(rows[0]["note"], "保留,此列")
        self.assertEqual(self.headers, list(rows[0]))
        self.assertIs(answer_node.step_5_call_llm_create_answer, original_call)
        answer_call.assert_not_called()
        answer_save.assert_not_called()
        user_save.assert_not_called()

    def test_missing_prompt_does_not_write_outputs(self):
        answer_node = SimpleNamespace(step_5_call_llm_create_answer=lambda *args: ("", {}),
                                      step_7_save_assistant_message=lambda state: None)
        item_node = SimpleNamespace(step_7_save_user_chat_message=lambda state: None)
        app = FakeApp(answer_node, item_node, docs=[])
        args = self.args()
        with self.assertRaisesRegex(RuntimeError, "RAG 采集失败"):
            sft_data.write_outputs(self.val, args.val_jsonl, args.val_output_csv,
                                   args, app, lambda **kwargs: kwargs, answer_node, item_node)
        self.assertFalse(args.val_jsonl.exists())
        self.assertFalse(args.val_output_csv.exists())

        app.invoke = lambda state: {"all_docs": [{"text": "有证据，但没有回答调用"}]}
        with self.assertRaisesRegex(RuntimeError, "未调用模型一次"):
            sft_data.write_outputs(self.val, args.val_jsonl, args.val_output_csv,
                                   args, app, lambda **kwargs: kwargs, answer_node, item_node)
        self.assertFalse(args.val_jsonl.exists())
        self.assertFalse(args.val_output_csv.exists())

    def test_failure_keeps_checkpoint_and_rerun_resumes(self):
        with self.train.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.headers).writerow({
                "sample_id": "2", "query": "问题二", "input": "",
                "target/ground_truth": "答案二", "note": "第二行",
            })
        answer_node = SimpleNamespace(step_5_call_llm_create_answer=lambda *args: ("", {}),
                                      step_7_save_assistant_message=lambda state: None)
        item_node = SimpleNamespace(step_7_save_user_chat_message=lambda state: None)
        args = self.args()
        calls = []

        def fail_on_second(state):
            calls.append(state["original_query"])
            if state["original_query"] == "问题二":
                raise RuntimeError("临时故障")
            answer_node.step_5_call_llm_create_answer(
                f"规则\n参考资料：检索内容\n问题：{state['original_query']}", False, state["session_id"])
            return {"all_docs": [{"text": "检索内容"}]}

        app = SimpleNamespace(invoke=fail_on_second)
        with self.assertRaisesRegex(RuntimeError, "已保存 1 条"):
            sft_data.write_outputs(self.train, args.train_jsonl, args.train_output_csv,
                                   args, app, lambda **kwargs: kwargs, answer_node, item_node)
        checkpoint = Path(str(args.train_jsonl) + ".checkpoint.json")
        partial = Path(str(args.train_jsonl) + ".partial")
        partial_csv = Path(str(args.train_output_csv) + ".partial")
        self.assertTrue(checkpoint.exists())
        self.assertEqual(len(partial.read_text(encoding="utf-8").splitlines()), 1)
        with partial_csv.open(encoding="utf-8", newline="") as handle:
            self.assertEqual([row["query"] for row in csv.DictReader(handle)], ["问题一"])
        self.assertFalse(args.train_jsonl.exists())
        self.assertFalse(args.train_output_csv.exists())

        original = self.train.read_bytes()
        self.train.write_bytes(original.replace("问题二".encode(), "改动了".encode()))
        with self.assertRaisesRegex(ValueError, "输入 CSV 已变化"):
            sft_data.write_outputs(self.train, args.train_jsonl, args.train_output_csv,
                                   args, app, lambda **kwargs: kwargs, answer_node, item_node)
        self.train.write_bytes(original)

        def succeed(state):
            calls.append(state["original_query"])
            answer_node.step_5_call_llm_create_answer(
                f"规则\n参考资料：检索内容\n问题：{state['original_query']}", False, state["session_id"])
            return {"all_docs": [{"text": "检索内容"}]}
        app.invoke = succeed
        count = sft_data.write_outputs(self.train, args.train_jsonl, args.train_output_csv,
                                       args, app, lambda **kwargs: kwargs, answer_node, item_node)
        self.assertEqual(count, 2)
        self.assertEqual(calls, ["问题一", "问题二", "问题二"])
        self.assertFalse(checkpoint.exists())
        self.assertFalse(partial.exists())
        self.assertFalse(partial_csv.exists())
        self.assertEqual(len(args.train_jsonl.read_text(encoding="utf-8").splitlines()), 2)

    def test_no_subject_uses_query_and_clarification_in_both_outputs(self):
        answer_call = Mock(side_effect=AssertionError("不应调用回答模型"))
        answer_node = SimpleNamespace(step_5_call_llm_create_answer=answer_call,
                                      step_7_save_assistant_message=Mock())
        item_node = SimpleNamespace(step_7_save_user_chat_message=Mock())
        args = self.args()
        original = self.train.read_bytes()

        def no_subject(state):
            item_node.step_7_save_user_chat_message(state)
            answer_node.step_7_save_assistant_message(state)
            return {"all_docs": [], "item_names": [], "answer": sft_data.NO_SUBJECT_ANSWER}

        sft_data.write_outputs(self.train, args.train_jsonl, args.train_output_csv,
                               args, SimpleNamespace(invoke=no_subject), lambda **kwargs: kwargs,
                               answer_node, item_node)
        with args.train_output_csv.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        record = json.loads(args.train_jsonl.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["input"], "问题一")
        self.assertEqual(row["target/ground_truth"], sft_data.NO_SUBJECT_ANSWER)
        self.assertEqual(record["messages"], [
            {"role": "user", "content": "问题一"},
            {"role": "assistant", "content": sft_data.NO_SUBJECT_ANSWER},
        ])
        self.assertEqual(self.train.read_bytes(), original)
        answer_call.assert_not_called()
        answer_node.step_7_save_assistant_message.assert_not_called()
        item_node.step_7_save_user_chat_message.assert_not_called()

    def test_empty_subject_and_docs_do_not_require_matching_answer(self):
        answer_node = SimpleNamespace(step_5_call_llm_create_answer=lambda *args: ("", {}),
                                      step_7_save_assistant_message=lambda state: None)
        item_node = SimpleNamespace(step_7_save_user_chat_message=lambda state: None)
        args = self.args()
        app = SimpleNamespace(invoke=lambda state: {
            "all_docs": [], "item_names": [], "answer": "其他提前返回",
        })
        sft_data.write_outputs(self.val, args.val_jsonl, args.val_output_csv,
                               args, app, lambda **kwargs: kwargs, answer_node, item_node)
        with args.val_output_csv.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        record = json.loads(args.val_jsonl.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["input"], "问题一")
        self.assertEqual(row["target/ground_truth"], sft_data.NO_SUBJECT_ANSWER)
        self.assertEqual(record["messages"][1]["content"], sft_data.NO_SUBJECT_ANSWER)

    def test_project_utils_package_precedes_script_utils(self):
        script_dir = str(SOURCE.parent)
        project_root = str(sft_data.PROJECT_ROOT)
        with patch.object(sys, "path", [script_dir, project_root, *sys.path]):
            sft_data.prioritize_project_imports()
            spec = importlib.machinery.PathFinder.find_spec("utils", sys.path)
            self.assertEqual(Path(spec.origin), sft_data.PROJECT_ROOT / "utils/__init__.py")

    def test_column_names_output_paths_and_overwrite_are_configurable(self):
        args = self.args("--overwrite-source", "--answer-column", "target/ground_truth",
                         "--messages-key", "conversations", "--session-prefix", "custom")
        self.assertEqual(args.train_output_csv, self.train)
        self.assertEqual(args.val_output_csv, self.val)
        self.assertEqual(args.train_jsonl, self.train.with_suffix(".jsonl"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--train-output-csv", str(self.val))


if __name__ == "__main__":
    unittest.main()
