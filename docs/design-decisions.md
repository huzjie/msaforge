# 设计决策记录（ADR 精简版）

每一条记录「为什么这样选、放弃了什么」。

## ADR-1：block 粒度而非 token 粒度

- **决定**：index 分支按 block（默认 64 token）打分，而非逐 token。
- **理由**：①块级选择更好并行、对齐 GPU 连续内存；②GQA 拆小 head 后，token 散点读会
  变成多次独立小读，块级连续读避免这个坑；③排序对象从 n 降到 n/block_size。
- **放弃**：token 级更细粒度的精确性（MiniMax 实测损失可忽略）。

## ADR-2：per-KV-group 独立选块

- **决定**：每个 KV head 组独立跑 top-k 选择。
- **理由**：GQA 的存在意义是让不同 head 组关注不同输入；共用选择会抵消多样性
  （DSA 移植 GQA 的已知失败模式）。
- **放弃**：跨组共享选择带来的少量计算节省。

## ADR-3：零依赖纯 Python

- **决定**：核心只用标准库，list-of-list 手写张量运算。
- **理由**：可复现、可单测、任何环境直接 `python -m msaforge` 跑；托管环境无 numpy/torch。
- **放弃**：性能（参考实现不追求真实 GPU 吞吐，性能用 FLOPs 成本模型表征）。

## ADR-4：skill 恒正 delta 训练

- **决定**：`skill = min(1.0, skill + delta_up)`，delta 恒正。
- **理由**：「对 +1 错 -1」在 skill 低时期望漂移为负、会把 skill 压到 0。
- **放弃**：用真实梯度下降（mock 世界模型没有可微损失）。
