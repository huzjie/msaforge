"""Tests for config loading and the YAML fallback parser."""
import os
import tempfile
import unittest

from msaforge.config import Config, load_config, _parse_yaml
from msaforge.utils.yamlish import parse


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = Config()
        self.assertEqual(cfg.attention.dim, 256)
        self.assertEqual(cfg.backend.name, "mock")

    def test_from_dict(self):
        cfg = Config.from_dict({"attention": {"dim": 128}, "backend": {"name": "cpu"}})
        self.assertEqual(cfg.attention.dim, 128)
        self.assertEqual(cfg.backend.name, "cpu")

    def test_load_default_yaml(self):
        cfg = load_config()
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.attention.dim, 256)

    def test_yamlish_block_list(self):
        d = parse("keywords:\n  - a\n  - b\n")
        self.assertEqual(d["keywords"], ["a", "b"])

    def test_yamlish_nested(self):
        d = parse("attention:\n  dim: 256\n  top_k: [8, 16]\n")
        self.assertEqual(d["attention"]["dim"], 256)
        self.assertEqual(d["attention"]["top_k"], [8, 16])

    def test_yamlish_inline_comment(self):
        d = parse("name: mock  # a comment\n")
        self.assertEqual(d["name"], "mock")


if __name__ == "__main__":
    unittest.main()
