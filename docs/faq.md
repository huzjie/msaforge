# FAQ

**Q: 这是 MiniMax M3 的官方代码吗？**
不是。这是 MSA 稀疏注意力思想的独立零依赖参考实现，用于教学、单测和工程参考，与
MiniMax 官方无关。

**Q: 能直接跑真实 MiniMax M3 权重吗？**
不能。本仓库是参考实现，不含权重。要跑真实模型请用 MiniMax 官方权重（428B MoE），
本仓库的 `backends/` 里的 openai/vllm/transformers 后端可对接真实服务或权重。

**Q: 为什么是纯 Python、零依赖？**
为了「任何环境都能跑、都能单测、都能读懂」。参考实现的价值在思路可复用，不在吞吐。

**Q: prefill 9x / decode 15x 是怎么来的？**
FLOPs 成本模型下的理论加速比 = n_blocks / top_k。取 top_k ≈ n_blocks/9 与 n_blocks/15
即复现 MiniMax 报告的量级。真实硬件还会受内存带宽、kernel 实现影响。

**Q: mock 后端的 skill 是什么？**
它把「index 分支选块能力」抽象成一个标量，训练目标是从 0.5 爬到 1.0。它真正参与打分，
所以训练信号干净、召回率单调。

**Q: 怎么接自己的真实模型？**
改 `configs/default.yaml` 的 `backend.name`，或通过 `/v1/chat/completions` 的 OpenAI
兼容接口把 `base_url` 指到本服务。
