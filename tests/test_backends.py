"""Tests for the five backends and the registry."""
import unittest

from msaforge.config import Config
from msaforge.backends import list_backends, get_backend


class TestBackends(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_all_registered(self):
        names = list_backends()
        for expected in ["mock", "cpu", "openai", "vllm", "transformers"]:
            self.assertIn(expected, names)

    def test_select_and_generate(self):
        for name in list_backends():
            b = get_backend(name, self.cfg)
            sel, golds = b.select_blocks("q", 32, top_k=4)
            self.assertGreaterEqual(len(sel), 1)
            out = b.generate("hello", max_tokens=8)
            self.assertIsInstance(out, str)
            self.assertTrue(len(out) > 0)

    def test_mock_train_step_monotonic(self):
        b = get_backend("mock", self.cfg)
        s0 = b.skill
        for _ in range(5):
            s1 = b.train_step()
            self.assertGreater(s1, s0)
            s0 = s1
        self.assertLessEqual(b.skill, 1.0 + 1e-9)

    def test_mock_recall_high_skill(self):
        b = get_backend("mock", self.cfg)
        b.skill = 1.0
        sel, golds = b.select_blocks("q", 32, top_k=4)
        self.assertAlmostEqual(b.recall(sel, golds), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
