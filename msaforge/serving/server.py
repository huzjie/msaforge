"""OpenAI-compatible HTTP server built on stdlib http.server.

No Flask/FastAPI/uvicorn needed. Endpoints:

    GET  /health                -> {"status": "ok"}
    GET  /v1/models             -> model list
    POST /v1/chat/completions   -> OpenAI chat completion
    POST /v1/select             -> sparse block selection for a query
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ..backends import get_backend
from ..utils.logging import get_logger

log = get_logger("msaforge.serving")


def _json(obj, code: int = 200):
    body = json.dumps(obj).encode("utf-8")
    return code, body


class MSARequestHandler(BaseHTTPRequestHandler):
    backend = None
    cfg = None

    def log_message(self, fmt, *args):  # silence default stderr logging
        return

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok", "service": "msaforge"})
        elif self.path == "/v1/models":
            self._send({"object": "list", "data": [
                {"id": self.cfg.backend.model_id, "object": "model",
                 "owned_by": "msaforge"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = self._read_json()
        if self.path == "/v1/chat/completions":
            prompt = ""
            msgs = body.get("messages", [])
            if msgs:
                prompt = msgs[-1].get("content", "")
            out = self.backend.generate(prompt, max_tokens=body.get("max_tokens", 128))
            self._send({
                "id": "chatcmpl-msaforge", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant",
                                                     "content": out},
                             "finish_reason": "stop"}],
            })
        elif self.path == "/v1/select":
            q = body.get("query", "q0")
            n_blocks = int(body.get("n_blocks", 32))
            top_k = int(body.get("top_k", self.cfg.attention.top_k_blocks))
            sel, golds = self.backend.select_blocks(q, n_blocks, top_k=top_k)
            self._send({"query": q, "selected_blocks": sel, "gold_blocks": golds})
        else:
            self._send({"error": "not found"}, 404)


class MSAHTTPServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = get_backend(cfg.backend.name, cfg)

    def serve(self, host: str = "127.0.0.1", port: int = 8000):
        handler = type("Bound", (MSARequestHandler,),
                       {"backend": self.backend, "cfg": self.cfg})
        httpd = ThreadingHTTPServer((host, port), handler)
        log.info("serving msaforge on http://%s:%d", host, port)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()


def run_server(cfg, host: str = "127.0.0.1", port: int = 8000):
    MSAHTTPServer(cfg).serve(host, port)
