from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from worker.config import load

_KEYS = (
    "WORKER_ENGINE",
    "WORKER_MODEL",
    "WORKER_DEVICE",
    "WORKER_BLOCK_POOL",
    "WORKER_NUM_BLOCKS",
)


class ConfigLoadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ.pop(key, None) for key in _KEYS}

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_production_defaults_to_torch(self) -> None:
        cfg = load()
        self.assertEqual(cfg.engine_kind, "torch")
        self.assertEqual(cfg.model_id, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(cfg.device, "cuda")

    def test_engine_kind_override(self) -> None:
        os.environ["WORKER_ENGINE"] = "fake"
        cfg = load()
        self.assertEqual(cfg.engine_kind, "fake")

    def test_rejects_bad_engine(self) -> None:
        os.environ["WORKER_ENGINE"] = "vllm"
        with self.assertRaises(ValueError):
            load()
