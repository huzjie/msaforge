"""Tests for the sparse transformer and native multimodal model."""
import unittest

from msaforge.config import Config
from msaforge.model.transformer import SparseTransformer
from msaforge.model.vision import ViT
from msaforge.model.multimodal import NativeMultimodalModel


class TestTransformer(unittest.TestCase):
    def test_forward(self):
        cfg = Config()
        m = SparseTransformer(dim=32, n_layers=2, vocab_size=256,
                              block_size=8, top_k_blocks=2)
        logits = m.logits([1, 2, 3, 4, 5, 6, 7, 8], skill=1.0, query_key="q")
        self.assertEqual(logits.shape, (8, 256))


class TestVision(unittest.TestCase):
    def test_vit_encode(self):
        v = ViT(patch=16, dim=32, n_frames=2)
        frames = ViT.synthetic_frames(h=32, w=32, n=2)
        tok = v.encode(frames)
        self.assertGreater(tok.shape[0], 0)
        self.assertEqual(tok.shape[1], 32)


class TestMultimodal(unittest.TestCase):
    def test_fuse(self):
        cfg = Config()
        m = NativeMultimodalModel(dim=32, n_layers=2, vocab_size=256,
                                  patch=16, n_frames=2,
                                  block_size=8, top_k_blocks=2)
        frames = ViT.synthetic_frames(h=32, w=32, n=2)
        out = m.forward([1, 2, 3, 4], frames, skill=1.0, query_key="q")
        self.assertEqual(out.shape[1], 32)


if __name__ == "__main__":
    unittest.main()
