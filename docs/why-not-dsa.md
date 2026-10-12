# 为什么不用 DeepSeek 的 DSA：MQA 与 GQA 的差异

MiniMax 在公开分享里明确说，DeepSeek 的 DSA「非常强，但它是为 DeepSeek 自己的架构
（MQA + 高头维度）重度优化的」，把它硬移植到今天主流的 GQA 架构上会坏在两处。理解这两处，
比记住任何公式都更有复用价值。

## 背景：MQA vs GQA

- **MQA（Multi-Query Attention）**：所有 query head 共享**一个** KV head。
- **GQA（Grouped-Query Attention）**：query head 分成若干组，每组共享一个 KV head，
  在「省 KV cache」和「保持表达能力」之间折中。

## 失败模式一：丢失 KV 多样性

GQA 存在的意义，就是让**不同的 head 组能关注输入的不同部分**。如果稀疏注意力复用了
同一套 token 选择（所有组都盯同一批被选中的 token），那么：

> 「多个 KV 组最终都去关注同一批选中的 token，这等于悄悄抵消了 GQA 本该提供的多样性。」

换句话说：你为了省 KV cache 上了 GQA，结果稀疏化又把它的表达能力打了回去。

**解法（本仓库 `attention/gqa.py`）**：每个 KV head 组独立跑一次 block 选择，各自选
自己的 top-k。这样组与组之间仍然可以关注不同的 block。

```python
from msaforge.attention.gqa import select_per_kv_group
selections, golds = select_per_kv_group("query", n_blocks=32, n_kv_heads=2,
                                        top_k=4, skill=1.0)
# selections[0] 和 selections[1] 是两组各自独立的 top-k
```

## 失败模式二：GPU 内存访问效率下降

MQA 的一个大 KV head 是「一次连续读」；GQA 把它拆成几个小 head 后，如果按 token 散点
读取，就变成「多次独立的小读」，compute-to-memory 比下降，真实硬件上更慢。

**解法**：block 级连续读（一次读一个 block），把散点读重新变成连续读。这也是 MSA 坚持
block 粒度的硬件动因。

## 失败模式三（附带）：Top-K 的内部数据依赖

token 级 top-k 选择有内部依赖、难并行；block 级选择把排序对象从 n 个 token 降到
n/block_size 个 block，排序代价和依赖都大幅缩小。

## 一句话总结

**DSA 是对的思路，但它的实现假设了 MQA；换到 GQA 必须重新解决「多样性」和「内存访问」
两个问题。MSA 的做法是：per-KV-group 独立选块 + block 级连续读。**
