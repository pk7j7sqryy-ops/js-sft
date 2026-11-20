import importlib.util
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("sft.py")
SPEC = importlib.util.spec_from_file_location("sft_config_test", SOURCE)
import sys
sft = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sft
SPEC.loader.exec_module(sft)


class ConfigParsingTests(unittest.TestCase):
    def load_config(self, flags):
        values = {
            "MODEL_NAME_OR_PATH": "model",
            "FINETUNING_TYPE": "lora",
            "LORA_RANK": "16",
            "LEARNING_RATE": "0.0001",
            "NUM_TRAIN_EPOCHS": "1",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "1",
            "GRADIENT_ACCUMULATION_STEPS": "1",
            "LR_SCHEDULER_TYPE": "constant",
            "WARMUP_RATIO": "0",
            "EVAL_STEP_RATIO": "1",
            "CUTOFF_LEN": "2048",
            "TRAIN_DATA_FILE": "train.jsonl",
            "VAL_DATA_FILE": "val.jsonl",
            "WEIGHTS_NAME": "test",
            "EARLY_STOPPING_PATIENCE": "0",
            "GRAD_CLIP_NORM": "1",
            "WEIGHT_DECAY": "0",
            **flags,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("\n".join(f"{key}={value}" for key, value in values.items()))
            return sft.step_1_load_config(path)

    def test_boolean_flags_and_seed_are_converted(self):
        config = self.load_config({
            "TRAIN_MLP": "False", "TRAIN_ATTN": "True",
            "TRAIN_UNEMBED": "False", "SEED": "123",
        })
        self.assertIs(config.train_mlp, False)
        self.assertIs(config.train_attn, True)
        self.assertIs(config.train_unembed, False)
        self.assertEqual(config.seed, 123)

    def test_optional_flags_default_to_sdk_defaults(self):
        config = self.load_config({})
        self.assertIs(config.train_mlp, False)
        self.assertIs(config.train_attn, True)
        self.assertIs(config.train_unembed, False)
        self.assertEqual(config.seed, 42)

    def test_invalid_boolean_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "TRAIN_UNEMBED"):
            self.load_config({"TRAIN_UNEMBED": "sometimes"})


if __name__ == "__main__":
    unittest.main()
