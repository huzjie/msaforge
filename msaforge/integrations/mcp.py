"""Model Context Protocol (MCP) stdio server for msaforge.

Exposes two tools over stdio JSON-RPC:
  - msaforge_generate : generate text for a prompt
  - msaforge_select   : sparse block selection for a query

Run:  python -m msaforge.integrations.mcp
"""
import json
import sys

from ..config import load_config
from ..backends import get_backend

TOOLS = [
    {
        "name": "msaforge_generate",
        "description": "Generate text with the msaforge sparse-attention model backend.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "max_tokens": {"type": "integer", "default": 128},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "msaforge_select",
        "description": "Select sparse attention blocks for a query (index branch).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "n_blocks": {"type": "integer", "default": 32},
                "top_k": {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    },
]


def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _dispatch(cfg, method, params):
    backend = get_backend(cfg.backend.name, cfg)
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "msaforge_generate":
            out = backend.generate(args.get("prompt", ""),
                                   max_tokens=args.get("max_tokens", 128))
            return {"content": [{"type": "text", "text": out}]}
        if name == "msaforge_select":
            sel, golds = backend.select_blocks(args.get("query", "q0"),
                                               int(args.get("n_blocks", 32)),
                                               top_k=int(args.get("top_k", 8)))
            return {"content": [{"type": "text", "text": json.dumps(
                {"selected_blocks": sel, "gold_blocks": golds})}]}
        raise ValueError("unknown tool: %s" % name)
    raise ValueError("unknown method: %s" % method)


def main():
    cfg = load_config()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params", {})
        try:
            result = _dispatch(cfg, method, params)
            _send({"jsonrpc": "2.0", "id": rid, "result": result})
        except Exception as e:  # noqa: BLE001
            _send({"jsonrpc": "2.0", "id": rid,
                   "error": {"code": -32000, "message": str(e)}})


if __name__ == "__main__":
    main()
