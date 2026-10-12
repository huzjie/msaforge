"""Tests for training monotonicity and benchmark signals."""
import unittest

from msaforge.config import Config
from msaforge.backends.mock import MockBackend
from msaforge.train.trainer import Trainer
from msaforge.bench.recall import run_recall
from msaforge.bench.quality import make_signal_sequence, run_quality
from msaforge.bench.speed import run_speed, flops_speedup


class TestTrainer(unittest.TestCase):
    def test_skill_monotonic(self):
        cfg = Config()
        backend = MockBackend(cfg)
        trainer = Trainer(cfg, backend)
        hist = trainer.train(n_steps=10, n_blocks=32)
        skills = [s for s, _ in hist]
        self.assertEqual(skills, sorted(skills))
        self.assertAlmostEqual(backend.skill, 1.0, places=6)


class TestRecall(unittest.TestCase):
    def test_recall_monotonic(self):
        cfg = Config()
        points = run_recall(cfg, n_blocks=32, top_k=4, n_queries=20)
        recalls = [r for _, r in points]
        self.assertEqual(recalls, sorted(recalls))
        self.assertGreater(recalls[-1], recalls[0])


class TestQuality(unittest.TestCase):
    def test_quality_monotonic(self):
        cfg = Config()
        points = run_quality(cfg, seq_len=64, block_size=16, top_k=2, n_queries=10)
        cos = [c for _, c in points]
        self.assertEqual(cos, sorted(cos))
        self.assertGreater(cos[-1], cos[0])

    def test_signal_sequence(self):
        x = make_signal_sequence(64, 32, 2, 16)
        self.assertEqual(x.shape, (64, 32))


class TestSpeed(unittest.TestCase):
    def test_speedup_ratio(self):
        sp = run_speed()
        self.assertGreater(sp["prefill_speedup"], 5.0)
        self.assertGreater(sp["decode_speedup"], 10.0)

    def test_flops_speedup(self):
        self.assertAlmostEqual(flops_speedup(640, 64, 2), 5.0, places=3)


if __name__ == "__main__":
    unittest.main()
