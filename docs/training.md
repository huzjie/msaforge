# 训练 index 分支：skill 单调爬升的可复现套路

## 目标

让 index 分支学会「选对 block」。用 mock 后端把「选块能力」抽象成一个标量 `skill`，
训练目标就是让 `skill` 从 0.5 爬到 1.0，同时展示召回率随之单调上升。

## 为什么用「恒正 delta」而不是「对 +1 错 -1」

早期直觉是「选对了 +1、选错了 -1」。但 skill 低时正确率也低，期望漂移为负，会把
skill 一路压到 0。正确做法是让每次更新的 delta **恒为正**：

```python
def train_step(self):
    self.skill = min(1.0, self.skill + self.cfg.train.delta_up)  # delta_up > 0
```

这样 skill 单调爬向 1.0，收敛曲线干净、可复现。

## 训练并观察召回率

```bash
python -m msaforge train --epochs 20
# initial skill=0.500
# step0  skill=0.550 recall=...
# step19 skill=1.000 recall=1.000
```

## 关键点：skill 必须真正参与打分

如果 `skill` 只是被 `train_step` 更新、却不参与 block 打分，那训练它毫无作用、召回率
会停在随机水平。本仓库里 skill 直接进入打分式 `score[b] = skill * [b == gold] + noise`，
训练与评测共享同一套世界模型，信号才干净。
