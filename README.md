# msaforge — MiniMax Sparse Attention 零依赖参考实现

> 把 MiniMax M3 的 **MSA 稀疏注意力**（1M 上下文下 prefill 快 ~9x、decode 快 ~15x）
> 拆成两分支机制，用纯 Python（零依赖）做成可跑、可训练、可验证的完整框架。

[![CI](https://github.com/huzjie/msaforge/actions/workflows/ci.yml/badge.svg)](https://github.com/huzjie/msaforge/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

---

## 这个热点背后的技术原理

全注意力是 O(n²)：每个 query 都要看所有 key。在 1M token 上下文下，n² = 10¹²，这就是
「长上下文又贵又慢」的根源。MiniMax M3 的解法是 **MSA（MiniMax Sparse Attention）**，
把注意力拆成两条分支：

```
输入 x (seq_len × dim)
        │
        ▼
 ┌──────────────────────────────┐
 │ index 分支（轻量打分器）        │   O(n · rank)，rank ≪ dim
 │ · KV cache 按 block 切成块      │   （block_size=64，一次看一整个块）
 │ · 低秩投影给每个 block 打分     │
 └──────────────┬───────────────┘
                │ 保留 top-k 块
                ▼
 ┌──────────────────────────────┐
 │ sparse 分支（真实注意力）       │   只对选中的 top-k 块做
 │ scaled dot-product attention  │   softmax；其余 mask 成 -inf
 └──────────────┬───────────────┘
                ▼
            输出 (seq_len × dim)
```

**加速比 ≈ n_blocks / top_k**。取 top_k ≈ n_blocks/9 得 prefill ~9x，取 top_k ≈ n_blocks/15
得 decode ~15x。

### 两个关键设计决策（比公式更值钱）

1. **为什么选 block 而不是 token？** token 级稀疏在 GPU 上是散点读，GQA 拆小 head 后
   会变成「多次独立小读」、compute-to-memory 比下降；block 级一次读一段连续内存，更好
   并行、更好访存，且质量损失可忽略。
2. **为什么不用 DeepSeek 的 DSA？** DSA 为 MQA（所有 head 共享一个 KV）优化，移植到 GQA
   会坏两处：①所有 head 组盯同一批 token，抵消 GQA 的多样性；②散点读降低访存效率。
   MSA 的解法是**每个 KV head 组独立选块**。

---

## 可复用干货块

### ① 训练信号：skill 单调爬升的确定性打分

```python
# index 分支的 block 打分（skill 真正参与，且单调可验证）
def score_blocks(query_key, n_blocks, skill):
    gold = gold_block(query_key, n_blocks)          # 确定性「真正相关」块
    return [skill * (b == gold) + stable_float(f"noise:{query_key}:{b}")
            for b in range(n_blocks)]               # noise ∈ [0,1]
```

skill 0.5 → 1.0 时，gold block 从「和噪声打平」变成「恒最高」，召回率从随机爬到 1.0，
质量、加速、稀疏比全部随之单调。**这是全仓库最核心的可复现性设计**——skill 必须真正
参与打分，训练信号才干净。

### ② GQA 兼容：per-KV-group 独立选块

```python
for g in range(n_kv_heads):
    scores_g, gold_g = score_blocks(f"{query}|kv{g}", n_blocks, skill)
    selections[g] = top_k(scores_g, top_k)          # 每组独立，保住多样性
```

### ③ 可粘贴提示词（更多见 `prompts/`）

> 你是一位注意力机制专家。请用「index 分支（轻量打分）+ sparse 分支（只对选中块做真实
> attention）」的两分支框架讲稀疏注意力，解释为什么 block 粒度优于 token 粒度，并说明
> MQA 稀疏方案移植到 GQA 会丢失 KV 多样性这个坑及 per-KV-group 独立选块的解法。

---

## 快速开始（零依赖）

```bash
git clone https://github.com/huzjie/msaforge
cd msaforge

python -m msaforge doctor          # 校验五后端 + 组件
python -m msaforge train           # 训练 index 分支（skill 0.5 -> 1.0）
python -m msaforge bench           # speed / recall / quality 基准
python -m msaforge select "q0"     # 展示稀疏选块
python -m msaforge serve           # 启动 OpenAI 兼容 HTTP 服务
```

HTTP 服务：

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

---

## 基准（`python -m msaforge bench`）

| 基准 | 含义 | 结果 |
|---|---|---|
| **speed** | 1M 上下文 FLOPs 模型 | prefill ~9x / decode ~15x |
| **recall** | 选块命中关键块 | skill 0.5→1.0，召回随机→1.0 |
| **quality** | sparse vs full 余弦相似度 | 随 skill 单调 → 1.0 |

---

## 架构

```
msaforge/
  attention/    index 分支 + sparse 分支 + GQA + MSA 层   ← 核心
  model/        稀疏 Transformer + ViT(3D attention) + 原生多模态
  backends/     mock(可训练) / cpu / openai / vllm / transformers
  train/        index 分支 block 选择训练
  bench/        speed / recall / quality
  serving/      OpenAI 兼容 stdlib HTTP
  cli/          doctor / train / bench / select / serve
  integrations/ LangChain + MCP
docs/           技术拆解（architecture / msa / why-not-dsa / multimodality ...）
prompts/        4 个可粘贴提示词
recipes/        5 个分步配方
```

更多技术拆解见 [`docs/architecture.md`](docs/architecture.md)、
[`docs/msa-how-it-works.md`](docs/msa-how-it-works.md)、
[`docs/why-not-dsa.md`](docs/why-not-dsa.md)。

---

## License

Apache 2.0。本项目是 MiniMax M3 MSA 稀疏注意力的**独立参考实现**，与 MiniMax 官方无关。
