# 配方 03：训练 index 分支的 block 选择

## 目标
让 index 分支选块更准，召回率从随机爬到 1.0。

## 步骤

1. **把选块能力抽象成标量 skill**，让它真正参与打分：
   ```python
   score[b] = skill * (b == gold_block) + noise_b   # noise 确定性 ∈ [0,1]
   ```
2. **用恒正 delta 训练**（别用「对+1错-1」，会被压到 0）：
   ```python
   skill = min(1.0, skill + delta_up)
   ```
3. **每步记录召回**，观察 (skill, recall) 单调曲线。
4. **收敛判据**：skill=1.0 时召回恒 1.0，且曲线无回落。

## 验证
```bash
python -m msaforge train --epochs 20
```
