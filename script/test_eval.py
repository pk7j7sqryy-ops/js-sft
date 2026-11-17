import importlib.util
import math
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).with_name("eval.py")
SPEC = importlib.util.spec_from_file_location("ragas_eval_script", SOURCE)
eval_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_script)


class EvalScoringTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "question": "问题", "ground_truth": "标准答案", "answer": "回答",
            "contexts": ["检索内容"], "context": "[1] 检索内容",
        }

    def test_retries_only_missing_metrics_and_preserves_zero(self):
        calls = []

        def evaluate(dataset, batch_size, timeout, max_workers, metric_names):
            calls.append(set(metric_names))
            if len(calls) == 1:
                return [{
                    "faithfulness": 0,
                    "answer_relevancy": 0.75,
                    "context_precision": math.nan,
                    "context_recall": 0.5,
                    "answer_correctness": 0.9,
                }]
            self.assertEqual(metric_names, {"context_precision"})
            return [{"context_precision": 0.25}]

        with (patch.object(eval_script, "step_3_build_ragas_dataset", return_value=object()),
              patch.object(eval_script, "step_4_run_ragas_evaluation", side_effect=evaluate)):
            result = eval_script.evaluate_row_with_retries(
                self.row, timeout=10, max_workers=1, metric_retries=2
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(result["faithfulness"], 0.0)
        self.assertEqual(result["Context Precision"], 0.25)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["error"], "")

    def test_no_context_skips_context_metrics_and_keeps_valid_zero_scores(self):
        self.row["contexts"] = []
        self.row["context"] = ""

        def evaluate(dataset, batch_size, timeout, max_workers, metric_names):
            self.assertEqual(metric_names, {"answer_relevancy", "answer_correctness"})
            return [{"answer_relevancy": 0, "answer_correctness": 0.4}]

        with (patch.object(eval_script, "step_3_build_ragas_dataset", return_value=object()),
              patch.object(eval_script, "step_4_run_ragas_evaluation", side_effect=evaluate)):
            result = eval_script.evaluate_row_with_retries(
                self.row, timeout=10, max_workers=1, metric_retries=0
            )

        self.assertEqual(result["Answer Relevancy"], 0.0)
        self.assertEqual(result["Answer Correctness"], 0.4)
        self.assertEqual(result["faithfulness"], "")
        self.assertEqual(result["status"], "evaluation_partial")
        self.assertIn("无检索上下文", result["error"])


if __name__ == "__main__":
    unittest.main()
