"""Native multimodal fusion + 3D attention demo."""
import _bootstrap  # noqa: F401
from msaforge.model.multimodal import NativeMultimodalModel
from msaforge.model.vision import ViT

m = NativeMultimodalModel(dim=64, n_layers=2, vocab_size=1024,
                          patch=16, n_frames=2, block_size=8, top_k_blocks=2)
frames = ViT.synthetic_frames(h=32, w=32, n=2)
out = m.forward([1, 2, 3, 4], frames, skill=1.0, query_key="q")
print("fused sequence shape:", out.shape)
print("text + vision fused from token zero into one sparse transformer.")
