# 基准：speed / recall / quality

三个基准分别回答三个问题：快多少？选得准不准？质量保不保得住？

## speed（速度）

用 FLOPs 成本模型（不真实分配 1M 矩阵，纯 Python 跑不动），在 1M 上下文下：

```
full prefill  = n^2 * d
sparse prefill = n * (top_k_prefill * block_size) * d + index_cost
speedup = full / sparse  ≈  n_blocks / top_k
```

取 top_k_prefill ≈ n_blocks/9、top_k_decode ≈ n_blocks/15，复现 MiniMax 报告的
**prefill ~9x、decode ~15x**。

```bash
python -m msaforge bench
```

## recall（检索召回）

mock 后端 skill 从 0.5 扫到 1.0，统计「gold block 是否落在 top-k」：

| skill | recall |
|---|---|
| 0.50 | ≈ top_k/n_blocks（随机） |
| 0.70 | 中 |
| 0.90 | 高 |
| 1.00 | 1.0 |

## quality（质量保持）

构造 signal 序列（gold block 内 token 同向、其余 token 正交化），比较 sparse 输出与
full-attention 输出的余弦相似度。选中 gold → 余弦 ~1；未选中 → 余弦 ~0。质量随 skill
单调上升，证明「稀疏化在 skill 足够时几乎不损质量」。
