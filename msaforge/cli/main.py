"""Command-line interface.

    python -m msaforge doctor        # verify backends + components
    python -m msaforge train         # train the index branch (skill -> 1.0)
    python -m msaforge bench         # run speed/recall/quality benchmarks
    python -m msaforge select Q      # show sparse block selection for a query
    python -m msaforge serve         # start the OpenAI-compatible HTTP server
"""
import argparse
import sys

from ..config import load_config
from ..utils.logging import get_logger

log = get_logger("msaforge.cli")


def _common():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", default=None, help="path to config YAML")
    return p


def cmd_doctor(cfg, args):
    from ..backends import get_backend, list_backends
    print("backends: %s" % ", ".join(list_backends()))
    for name in list_backends():
        try:
            b = get_backend(name, cfg)
            sel, golds = b.select_blocks("doctor", 32, top_k=4)
            out = b.generate("hello", max_tokens=8)
            print("  [ok] %-14s select=%d blocks  generate=%r"
                  % (name, len(sel), out[:24]))
        except Exception as e:
            print("  [FAIL] %-14s %s" % (name, e))
    from ..attention.msa import MSALayer
    layer = MSALayer(cfg.attention.dim, block_size=cfg.attention.block_size,
                     top_k_blocks=cfg.attention.top_k_blocks)
    print("  [ok] MSALayer dim=%d block=%d top_k=%d"
          % (cfg.attention.dim, cfg.attention.block_size,
             cfg.attention.top_k_blocks))
    from ..model.transformer import SparseTransformer
    m = SparseTransformer(cfg.attention.dim, n_layers=2,
                          vocab_size=cfg.model.vocab_size)
    print("  [ok] SparseTransformer n_layers=2")
    from ..model.vision import ViT
    v = ViT(patch=cfg.model.vision_patch, dim=cfg.attention.dim)
    frames = ViT.synthetic_frames(n=2)
    tok = v.encode(frames)
    print("  [ok] ViT encode %d vision tokens" % tok.shape[0])
    print("doctor done.")


def cmd_train(cfg, args):
    from ..backends.mock import MockBackend
    from ..train.trainer import Trainer
    backend = MockBackend(cfg)
    trainer = Trainer(cfg, backend)
    print("initial skill=%.3f" % backend.skill)
    hist = trainer.train(n_steps=args.epochs, n_blocks=args.n_blocks)
    first = hist[0]
    last = hist[-1]
    print("step0  skill=%.3f recall=%.3f" % first)
    print("step%d skill=%.3f recall=%.3f" % (len(hist) - 1, last[0], last[1]))
    print("train done: skill %.3f -> %.3f" % (first[0], last[0]))


def cmd_bench(cfg, args):
    from ..bench.summary import run_all
    run_all(cfg)


def cmd_select(cfg, args):
    from ..backends import get_backend
    backend = get_backend(cfg.backend.name, cfg)
    sel, golds = backend.select_blocks(args.query, args.n_blocks,
                                       top_k=args.top_k)
    print("query=%r  selected_blocks=%s  gold_blocks=%s"
          % (args.query, sel, golds))


def cmd_serve(cfg, args):
    from ..serving.server import run_server
    run_server(cfg, host=args.host, port=args.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="msaforge",
                                 description="MiniMax Sparse Attention reference implementation")
    sub = ap.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", parents=[_common()], help="verify components")
    p_doctor.set_defaults(func=cmd_doctor)

    p_train = sub.add_parser("train", parents=[_common()], help="train index branch")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--n-blocks", type=int, default=32)
    p_train.set_defaults(func=cmd_train)

    p_bench = sub.add_parser("bench", parents=[_common()], help="run benchmarks")
    p_bench.set_defaults(func=cmd_bench)

    p_select = sub.add_parser("select", parents=[_common()], help="show block selection")
    p_select.add_argument("query")
    p_select.add_argument("--n-blocks", type=int, default=32)
    p_select.add_argument("--top-k", type=int, default=8)
    p_select.set_defaults(func=cmd_select)

    p_serve = sub.add_parser("serve", parents=[_common()], help="start HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
