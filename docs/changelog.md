# Changelog

## 1.0.0 (2026-10-12)

- 首次发布：MSA 稀疏注意力零依赖参考实现。
- index 分支（低秩打分）+ sparse 分支（top-k 块 softmax）。
- block 粒度切块 + per-KV-group 独立选块（GQA 兼容）。
- 确定性可训练 mock（skill 0.5→1.0，召回/质量/稀疏比单调）。
- 五后端（mock/cpu/openai/vllm/transformers）。
- speed（9x/15x）/ recall / quality 三基准。
- OpenAI 兼容 stdlib HTTP + CLI + LangChain/MCP 集成。
- Docker/K8s/Helm/CI + docs + prompts + recipes。
