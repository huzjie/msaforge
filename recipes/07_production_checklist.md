# 配方 07：稀疏注意力上线检查清单

上线一个稀疏注意力推理系统前，逐项确认：

- [ ] prefill/decode 加速比符合预期（本仓库 speed 基准）。
- [ ] 选块召回率可验证（recall 基准，skill→1 时 →1.0）。
- [ ] 质量保留率随 skill 单调（quality 基准）。
- [ ] GQA 组间多样性未丢失（per-KV-group 独立选块）。
- [ ] KV cache 淘汰与稀疏选择粒度一致（均为 block）。
- [ ] 后端 fallback 路径不联网也能返回（doctor 全绿）。
- [ ] OpenAI 兼容端点 `/health`、`/v1/chat/completions` 冒烟通过。
