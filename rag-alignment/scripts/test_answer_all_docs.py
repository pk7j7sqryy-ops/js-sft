import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[3] / "processor/query_processor/nodes/node_answer_output.py"


def load_answer_functions(**dependencies):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "node_answer_output", "step_4_create_answer_prompt"
        }:
            node.decorator_list = []
            node.returns = None
            for argument in node.args.args:
                argument.annotation = None
            functions.append(node)
    namespace = dict(dependencies)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class AnswerAllDocsTests(unittest.TestCase):
    def test_final_context_and_prompt_use_same_docs(self):
        reranked = [{"chunk_id": 1, "title": "精排", "text": "原始内容"},
                    {"chunk_id": 1, "title": "精排", "text": "原始内容"}]
        backfills = [{"chunk_id": 1, "text": "原始内容"},
                     {"chunk_id": 2, "title": "回填", "text": "相邻内容"}]
        create_prompt = Mock(return_value="提示词")
        namespace = load_answer_functions(
            QueryGraphState=dict,
            sys=SimpleNamespace(_getframe=lambda: SimpleNamespace(f_code=SimpleNamespace(co_name="node_answer_output"))),
            add_running_task=Mock(), add_done_task=Mock(),
            step_1_has_answer=lambda state: bool(state["answer"]),
            step_2_validate_and_get_data=lambda state: (state["reranked_docs"], "session", ["产品"], "问题", False),
            fill_adjacent_chunks=lambda docs, rrf: backfills,
            step_3_get_history_by_session_id=lambda session: [],
            step_5_call_llm_create_answer=lambda prompt, stream, session: ("回答", {}),
            step_6_extract_chunk_and_url_image=lambda docs: [],
            _normalize_usage=lambda usage: usage,
            step_7_save_assistant_message=Mock(),
            logger=SimpleNamespace(info=Mock()),
        )
        namespace["step_4_create_answer_prompt"] = create_prompt
        state = {"session_id": "session", "is_stream": False, "reranked_docs": reranked,
                 "rrf_chunks": [], "answer": ""}
        result = namespace["node_answer_output"](state)
        self.assertEqual([doc["chunk_id"] for doc in result["all_docs"]], [1, 2])
        self.assertIs(create_prompt.call_args.args[0], result["all_docs"])
        self.assertEqual(len(create_prompt.call_args.args), 4)

    def test_prompt_renders_context_only_once(self):
        render = Mock(side_effect=lambda name, **fields: fields)
        namespace = load_answer_functions(load_prompt=render, logger=SimpleNamespace(info=Mock()))
        output = namespace["step_4_create_answer_prompt"](
            [{"title": "精排", "text": "原始内容", "score": 0.9},
             {"title": "回填", "text": "相邻内容"}], [], ["产品"], "问题"
        )
        self.assertEqual(output["context"].count("原始内容"), 1)
        self.assertEqual(output["context"].count("相邻内容"), 1)
        self.assertNotIn("chunk_backfills", output)

    def test_preanswered_state_has_no_context(self):
        namespace = load_answer_functions(
            QueryGraphState=dict,
            sys=SimpleNamespace(_getframe=lambda: SimpleNamespace(f_code=SimpleNamespace(co_name="node_answer_output"))),
            add_running_task=Mock(), add_done_task=Mock(),
            step_1_has_answer=lambda state: bool(state["answer"]),
            step_7_save_assistant_message=Mock(),
        )
        result = namespace["node_answer_output"]({"session_id": "session", "answer": "已拒答", "all_docs": [{"text": "旧结果"}]})
        self.assertEqual(result["all_docs"], [])


if __name__ == "__main__":
    unittest.main()
