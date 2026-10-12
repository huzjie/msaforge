# 原生多模态：为什么 vision 要从 token zero 一起训练

MiniMax M3 的另一个关键选择：文本和视觉**从预训练第一步就一起训练**，而不是「先把文本
训好，再贴一个视觉 adapter」。MiniMax 报告后者会带来**不稳定的结果**。

## 传统做法：晚加 vision

```
文本模型（已训好）  --add vision adapter-->  多模态模型
```

问题：文本模型的 embedding 空间已经定型，视觉特征要被硬投影进一个为文本优化的空间，
训练过程容易震荡、还容易损伤原有的文本能力。

## MSA 的做法：从 token zero 融合

```
text tokens ──┐
              ├──> 同一个 token 空间 ──> 同一条稀疏 Transformer
vision tokens ─┘
```

- 文本 token 和视觉 token 投影到**同一个维度**，交织成一条序列，喂给同一条 MSA 骨干。
- 视觉塔内部用 **3D attention**（frame × y × x），把时间结构建模进来，而不是把视频
  帧拍平成 2D 网格丢掉时序。

## 可复用要点

1. **「晚融合」的代价不是算力，是稳定性**。贴 adapter 看着省事，实则引入一个新的
   投影层要训练，还会扰动已收敛的主干。
2. **统一 token 空间是「原生多模态」的地基**。任何模态进来，先变成同一空间的 token，
   后面所有层都不用改。
3. **3D attention 是视频理解的最小充分改造**——只把 key 的坐标从 (y, x) 扩成
   (frame, y, x)，就保留了时序。

参考实现见 `model/multimodal.py` 和 `model/vision.py`：

```python
from msaforge.model.multimodal import NativeMultimodalModel
from msaforge.model.vision import ViT

m = NativeMultimodalModel(dim=256, n_layers=4, vocab_size=4096)
frames = ViT.synthetic_frames(n=4)
out = m.forward([1, 2, 3, 4], frames, skill=1.0, query_key="q")
```
