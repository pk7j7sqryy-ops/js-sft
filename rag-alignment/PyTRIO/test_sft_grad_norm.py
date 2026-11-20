import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).with_name("sft.py")
SPEC = importlib.util.spec_from_file_location("sft_grad_norm", SOURCE)
sft = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = sft
SPEC.loader.exec_module(sft)


class GradNormTests(unittest.TestCase):
    def test_extracts_optimizer_metric(self):
        self.assertEqual(sft.step_9_extract_grad_norm(
            SimpleNamespace(metrics={"grad_norm": 2.5, "grad_clip_norm": 1.0})
        ), 2.5)
        self.assertEqual(sft.step_9_extract_grad_norm(
            {"metrics": {"gradient/norm": 0.75}}
        ), 0.75)

    def test_does_not_log_clip_threshold_or_invalid_metric(self):
        self.assertIsNone(sft.step_9_extract_grad_norm(
            SimpleNamespace(metrics={"grad_clip_norm": 1.0})
        ))
        self.assertIsNone(sft.step_9_extract_grad_norm(
            SimpleNamespace(metrics={"grad_norm": math.nan})
        ))
        self.assertIsNone(sft.step_9_extract_grad_norm(None))


if __name__ == "__main__":
    unittest.main()
