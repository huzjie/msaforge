# msaforge 架构

msaforge 是 MiniMax M3 的 MSA（MiniMax Sparse Attention）稀疏注意力的零依赖参考实现。
一句话概括它的设计哲学（Olive Song 原话）：**「更简单的架构，才有更好的 scaling 性质」**。

## 一句话原理

全注意力是 O(n^2)：每个 query token 都要看所有的 key token。MSA 把它拆成两条分支：

```
输入序列 x (seq_len x dim)
        |
        v
   ┌─────────────────────────────┐
   │  index 分支（轻量打分器）      │   O(n * rank)，rank << dim
   │  把 KV cache 按 block 切块     │
   │  给每个 block 打一个相关性分    │
   └──────────────┬──────────────┘
                  | top-k blocks
                  v
   ┌─────────────────────────────┐
   │  sparse 分支（真实注意力）     │   只对选中的 top-k blocks 做
   │  scaled dot-product attention│   softmax 注意力
   └──────────────┬──────────────┘
                  v
             输出 (seq_len x dim)
```

## 为什么是 block 粒度，而不是 token 粒度

历史上很多稀疏注意力（包括 DeepSeek 的 DSA）在 **token 粒度** 上选 token。MSA 改在
**block 粒度** 上选 block，一次选一整个块（比如 64 个 token）。这个改动看似小，带来三个收益：

1. **更好并行**：块级选择天然对齐 GPU 的连续内存布局，一次读一个块，而非散点读 token。
2. **更友好的内存访问**：GQA 把一个 512 维的大 KV 头拆成几个 128 维的小头，散点读会
   变成多次独立的小读，compute-to-memory 比下降；块级连续读能避免这个坑。
3. **几乎不损失质量**：MiniMax 内部测试表明块级选择相比 token 级选择的精度损失「可以忽略」。

## 本仓库的模块划分

| 模块 | 职责 | 对应 MSA 概念 |
|---|---|---|
| `attention/blocks.py` | KV 序列按固定块切分、块内均值 | block 划分 |
| `attention/index.py` | 低秩打分器 + 世界模型打分 | index 分支 |
| `attention/sparse.py` | 只对选中块做 softmax 注意力 | sparse 分支 |
| `attention/gqa.py` | 每组 KV head 独立选块 | GQA 兼容 |
| `attention/msa.py` | 组合 index + sparse 的完整层 | MSA 层 |
| `model/transformer.py` | 全 MSA 层的稀疏 Transformer | 骨干 |
| `model/vision.py` | patch 嵌入 + 3D 时空注意力 ViT | 视觉塔 |
| `model/multimodal.py` | 文本视觉从第一步融合 | 原生多模态 |
| `backends/` | mock/cpu/openai/vllm/transformers | 推理后端 |
| `train/` | index 分支块选择 skill 训练 | 训练 |
| `bench/` | speed/recall/quality 基准 | 评测 |
| `serving/` | OpenAI 兼容 stdlib HTTP | 服务 |
