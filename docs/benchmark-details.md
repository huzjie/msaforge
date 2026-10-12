# 基准细节

## speed 的 FLOPs 模型

```
full prefill   = n^2 * d
sparse prefill = n * (top_k_prefill * block_size) * d + n_blocks * rank
full decode    = n * d
sparse decode  = (top_k_decode * block_size) * d
speedup = full / sparse  ≈  n_blocks / top_k
```

在 1M token（n=1,048,576, block_size=64 → n_blocks=16384）：

- prefill：top_k_prefill = 1820 → speedup ≈ 9.0x
- decode：top_k_decode = 1092 → speedup ≈ 15.0x

## recall 的确定性世界模型

```python
gold = gold_block(query_key, n_blocks)      # md5 种子确定性
boost = (skill - 0.5) * 2.0                 # skill=0.5 -> 0, skill=1.0 -> 1.0
score[b] = boost * (b == gold) + noise_b    # noise 确定性 ∈ [0,1]
```

skill=0.5 时 selection 纯随机（recall ≈ top_k/n_blocks），skill=1.0 时 gold 恒最高
（recall=1.0）。单调、可复现。

## quality 的 signal 序列

gold block 内 token 全沿同一方向 `signal`，其余 token 与 signal 正交。quality =
sparse 输出在 signal 方向的投影 / full 输出的投影。选中 gold → 投影 ≈ full；未选中
→ 投影 ≈ 0。>1 表示稀疏化后 signal 更聚焦（排除了噪声块的注意力稀释）。
