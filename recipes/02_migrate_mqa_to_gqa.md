# 配方 02：从 MQA 稀疏方案迁移到 GQA

## 背景
DeepSeek 的 DSA 为 MQA + 高头维度优化。迁移到 GQA 会坏在两处，本配方给出解法。

## 步骤

1. **定位共享选择点**。找到「所有 head 共用一套 token/block 选择」的代码。
2. **改成 per-KV-group 独立选择**：
   ```python
   for g in range(n_kv_heads):
       scores_g, gold_g = score_blocks(f"{query}|kv{g}", n_blocks, skill)
       selections[g] = top_k(scores_g, top_k)
   ```
3. **把散点读改成 block 连续读**。GQA 拆小 head 后，按 block 一次读一段，避免多次独立小读。
4. **回归验证多样性**。断言不同组的 selections 不完全相同（skill 中间值时）。

## 验证
```python
from msaforge.attention.gqa import select_per_kv_group
sel, golds = select_per_kv_group("q", 32, 2, 4, 0.7)
assert len(set(map(tuple, sel))) > 1   # 组间多样性仍在
```
