# API 参考

## 核心类

| 类 / 函数 | 位置 | 说明 |
|---|---|---|
| `MSALayer` | `attention/msa.py` | 组合 index + sparse 的完整 MSA 层 |
| `IndexBranch` | `attention/index.py` | 低秩 block 打分器 |
| `SparseBranch` | `attention/sparse.py` | 只对选中块做 softmax 注意力 |
| `chunk_blocks` | `attention/blocks.py` | KV 序列按块切分 |
| `select_per_kv_group` | `attention/gqa.py` | per-KV-group 独立选块 |
| `SparseTransformer` | `model/transformer.py` | 全 MSA 层稀疏 Transformer |
| `ViT` / `Temporal3DAttention` | `model/vision.py` | patch 嵌入 + 3D 时空注意力 |
| `NativeMultimodalModel` | `model/multimodal.py` | token zero 多模态融合 |
| `get_backend` | `backends/__init__.py` | 后端注册表 |

## MSALayer

```python
MSALayer(dim, block_size=64, top_k_blocks=32, rank=32, seed="msa")
  .forward(x, query_key=None, skill=1.0, return_meta=False)
      # x: (seq_len, dim) Tensor
      # query_key: 提供则走世界模型打分（可训练），否则真实投影打分
      # return_meta=True 时返回 (output, meta)，meta 含 n_blocks/selected/gold/sparse_ratio
```

## 后端统一接口

```python
backend.select_blocks(query_key, n_blocks, n_kv_heads=1, top_k=None) -> (selections, golds)
backend.generate(prompt, max_tokens=128) -> str
backend.train_step(query_key="train") -> float   # 返回更新后的 skill
backend.recall(selections, golds) -> float        # mock 专用
```

## 配置（`configs/default.yaml`）

```yaml
attention:
  dim: 256          # 隐藏维度
  block_size: 64    # 每块 token 数
  top_k_blocks: 32  # 稀疏分支保留的块数
model:
  n_layers: 6
  vocab_size: 4096
backend:
  name: mock        # mock/cpu/openai/vllm/transformers
  skill: 0.5        # mock 选块能力（可训练）
```
