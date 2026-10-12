# 提示词：生成 MSA 稀疏注意力实现骨架

```
请为「MiniMax Sparse Attention (MSA)」写一个纯 Python（零依赖）参考实现骨架，用于教学和单测。

硬性要求：
1. 提供 index 分支（低秩打分器，rank 远小于 dim）和 sparse 分支（只对 top-k blocks 做 softmax）。
2. block 切分函数 chunk_blocks(seq_len, block_size) 返回 [(start, end), ...]。
3. GQA 兼容：每个 KV head 组独立选 top-k，返回 per-group 的选择列表。
4. 训练信号：block 打分 = skill * [b == gold_block] + noise（noise 确定性落在 [0,1]），
   保证 skill 0.5 -> 1.0 时召回率单调爬升。
5. 所有矩阵操作用 list-of-list 手写，不依赖 numpy/torch。
6. 给出对应 unittest，断言：skill=1.0 时 gold block 分数最高；召回率随 skill 单调。

代码风格：{{风格}}
