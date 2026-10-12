"""Train the index branch and watch recall climb."""
import _bootstrap  # noqa: F401
from msaforge.config import load_config
from msaforge.backends.mock import MockBackend
from msaforge.train.trainer import Trainer

cfg = load_config()
backend = MockBackend(cfg)
print("initial skill=%.3f" % backend.skill)
hist = Trainer(cfg, backend).train(n_steps=20, n_blocks=32)
print("step0  skill=%.3f recall=%.3f" % hist[0])
print("step%d skill=%.3f recall=%.3f" % (len(hist) - 1, hist[-1][0], hist[-1][1]))
