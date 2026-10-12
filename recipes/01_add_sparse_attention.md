# 配方 01：给现有 Transformer 加稀疏注意力

## 目标
把一个 full-attention Transformer 改造成 block 级稀疏注意力，长上下文提速。

## 步骤

1. **加 block 切分**。把 KV 序列按 `block_size`（建议 64）切成块：
   ```python
   blocks = [(s, min(s + block_size, n)) for s in range(0, n, block_size)]
   ```
2. **加 index 分支**。用低秩投影给每个 block 打分（rank 取 dim 的 1/8 左右），代价 O(n_blocks * rank)。
3. **选 top-k**。`top_k = n_blocks // 目标加速比`，例如要 9x 就取 `n_blocks // 9`。
4. **改 sparse 分支**。真实 attention 只对选中的 top-k blocks 做，其余 mask 成 -inf。
5. **验证召回**。用一个确定性的 gold block 检查「选中的块是否命中关键块」。

## 验证
```bash
python -m msaforge bench   # 看 speed / recall / quality
```

## 坑
- 别在 token 粒度选，散点读会让 GQA 的 GPU 内存访问效率崩掉。
- 别让所有 head 组共用同一套选择，会抵消 GQA 的多样性。
