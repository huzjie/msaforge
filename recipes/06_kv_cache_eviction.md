# 配方 06：用 block 分数做 KV cache 淘汰

## 步骤

1. 复用 index 分支的 block 分数，维护每个 block 的累计相关度。
2. 当 cache 超过预算时，淘汰分数最低的 block（而非逐 token）。
3. 用 `signal 保留率`（见 quality 基准）度量淘汰后的质量下降。
4. 验证：淘汰后 recall 仍随 skill 单调。

## 坑
- 别逐 token 淘汰（散点、慢、破坏块连续性）。
- 淘汰粒度 = block 粒度，和稀疏选择保持一致。
