"""Tests for block chunking, index branch, sparse branch, GQA, MSA layer."""
import unittest

from msaforge.attention.blocks import chunk_blocks, block_mean
from msaforge.attention.index import (
    IndexBranch, gold_block, score_blocks_with_skill, select_top_k,
)
from msaforge.attention.sparse import SparseBranch, full_attention, cosine
from msaforge.attention.gqa import group_heads, select_per_kv_group
from msaforge.attention.msa import MSALayer
from msaforge.core.tensor import Tensor


class TestBlocks(unittest.TestCase):
    def test_chunking(self):
        self.assertEqual(chunk_blocks(100, 64), [(0, 64), (64, 100)])

    def test_block_mean(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        means = block_mean(x, [(0, 2), (2, 3)])
        self.assertEqual(means[0], [2.0, 3.0])
        self.assertEqual(means[1], [5.0, 6.0])


class TestIndex(unittest.TestCase):
    def test_gold_deterministic(self):
        self.assertEqual(gold_block("q", 32), gold_block("q", 32))

    def test_score_skill_monotonic(self):
        # at skill=1 the gold block must have the highest score
        scores, gold = score_blocks_with_skill("q", 32, 1.0)
        self.assertEqual(max(range(len(scores)), key=lambda i: scores[i]), gold)

    def test_select_top_k(self):
        sel = select_top_k([0.1, 0.9, 0.5, 0.8], 2)
        self.assertEqual(sorted(sel), [1, 3])

    def test_index_real_scores(self):
        idx = IndexBranch(16, rank=8)
        q = [0.1] * 16
        breps = [[0.2] * 16 for _ in range(4)]
        scores = idx.real_scores(q, breps)
        self.assertEqual(len(scores), 4)


class TestSparse(unittest.TestCase):
    def test_sparse_shape(self):
        x = Tensor([[float(i + j) for j in range(8)] for i in range(16)])
        sp = SparseBranch(8)
        out = sp.forward(x, x, x, [(0, 4), (8, 12)])
        self.assertEqual(out.shape, (16, 8))

    def test_full_shape(self):
        x = Tensor([[0.1 * (i + j) for j in range(8)] for i in range(16)])
        out = full_attention(x, x, x)
        self.assertEqual(out.shape, (16, 8))

    def test_cosine_range(self):
        a = Tensor([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(cosine(a, a), 1.0, places=5)


class TestGQA(unittest.TestCase):
    def test_group_heads(self):
        groups = group_heads(8, 2)
        self.assertEqual(groups, [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_select_per_group(self):
        sel, golds = select_per_kv_group("q", 32, 2, 4, 1.0)
        self.assertEqual(len(sel), 2)
        self.assertEqual(len(golds), 2)
        for s, g in zip(sel, golds):
            self.assertIn(g, s)


class TestMSALayer(unittest.TestCase):
    def test_forward_shape(self):
        layer = MSALayer(16, block_size=8, top_k_blocks=2)
        x = Tensor([[0.1 * (i + j) for j in range(16)] for i in range(32)])
        out = layer.forward(x, query_key="q0", skill=1.0)
        self.assertEqual(out.shape, (32, 16))

    def test_meta_sparse_ratio(self):
        layer = MSALayer(16, block_size=8, top_k_blocks=2)
        x = Tensor([[0.1] * 16 for _ in range(32)])
        _, meta = layer.forward(x, query_key="q0", skill=1.0, return_meta=True)
        self.assertEqual(meta["n_blocks"], 4)
        self.assertEqual(meta["sparse_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
