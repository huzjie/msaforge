# MSA 怎么工作：从 index 分支到 sparse 分支

## 1. 全注意力的瓶颈

标准 scaled dot-product attention，对长度 n 的序列：

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

- **prefill**（一次喂入整段 prompt）：Q、K、V 都是 n 个 token，QK^T 是 n×n 矩阵，
  代价 O(n^2 · d)。
- **decode**（每生成一个新 token）：新 query 只 1 个，但要和全部 n 个历史 KV 相乘，
  代价 O(n · d)，且 KV cache 随 n 线性增长。

在 1M token 上下文下，n^2 = 10^12 量级，这就是「长上下文又贵又慢」的根源。

## 2. index 分支：把选择做得极便宜

index 分支用一个**低秩投影**给每个 block 打分。它不计算真实 attention，只做：

```
score(block_b) = dot(Wq · query_repr, Wk · block_repr_b) / sqrt(rank)
```

其中 `rank` 远小于 `d`（例如 d=256、rank=32）。每个 block 的代价是 O(rank)，
n 个 block 总代价 O(n · rank)，相比真实 attention 的 O(block_size · d) 便宜一个量级。

> **可复用要点**：index 分支的核心是「用一个便宜的代理信号，去预测哪个区域值得做贵的
> 真实计算」。这是所有「计算路由 / 早退 / 稀疏化」系统的通用套路——先廉价地筛，再贵地算。

## 3. sparse 分支：只对选中的 block 做真实 attention

选出 top-k 个 block 后，sparse 分支只对这些 block 里的 token 做真实 softmax 注意力，
其余 block 直接 mask 掉（score = -inf）。注意力的计算量从：

```
O(n^2 · d)   ->   O(n · (top_k · block_size) · d)
```

即加速比约等于 `n_blocks / top_k`。

## 4. 一个可复现的最小例子

```python
from msaforge.attention.msa import MSALayer
from msaforge.core.tensor import Tensor

layer = MSALayer(dim=256, block_size=64, top_k_blocks=32)
x = Tensor([[0.1 * (i + j) for j in range(256)] for i in range(2048)])
out, meta = layer.forward(x, query_key="my-query", skill=1.0, return_meta=True)
print("n_blocks:", meta["n_blocks"], "selected:", len(meta["selected"]))
print("sparse_ratio:", meta["sparse_ratio"])
```

## 5. 训练信号的设计（为什么 mock 可验证）

参考实现里 `skill` 直接参与 block 打分：

```
score[b] = skill * [b == gold_block] + noise_b     # noise 确定性落在 [0, 1]
```

- skill 低（0.5）时，gold block 和噪声打平，选块近似随机，召回率 ≈ top_k / n_blocks。
- skill 高（1.0）时，gold block 分数恒最高，召回率 → 1.0。

这条「单调召回」曲线驱动了质量、加速、稀疏比全部指标，是本仓库最核心的可复现性设计。
