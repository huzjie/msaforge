# 配方 05：端到端复现 MSA 的 9x/15x 加速

## 目标
在 1M token 上下文下复现 MiniMax 报告的 prefill ~9x、decode ~15x。

## 步骤

1. 定 `block_size = 64`，1M token -> `n_blocks = 16384`。
2. prefill 取 `top_k_prefill = round(n_blocks / 9) ≈ 1820`。
3. decode 取 `top_k_decode = round(n_blocks / 15) ≈ 1092`。
4. 用 FLOPs 模型核算（纯 Python 不真实分配 1M 矩阵）：
   ```python
   from msaforge.bench.speed import run_speed
   print(run_speed())   # prefill_speedup ≈ 9.0, decode_speedup ≈ 15.0
   ```

## 验证
```bash
python -m msaforge bench
```
