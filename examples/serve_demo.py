"""Start the HTTP server and smoke-test it end-to-end."""
import _bootstrap  # noqa: F401
import json
import threading
import time
from urllib import request

from msaforge.config import load_config
from msaforge.serving.server import MSAHTTPServer


def main():
    cfg = load_config()
    server = MSAHTTPServer(cfg)
    t = threading.Thread(target=server.serve, kwargs={"port": 8765}, daemon=True)
    t.start()
    time.sleep(0.5)

    base = "http://127.0.0.1:8765"
    h = request.urlopen(base + "/health", timeout=10)
    print("health:", h.read().decode())

    req = request.Request(base + "/v1/chat/completions",
                          data=json.dumps({"messages": [{"role": "user",
                                                         "content": "hi"}]}).encode(),
                          headers={"Content-Type": "application/json"})
    r = json.loads(request.urlopen(req, timeout=10).read().decode())
    print("completion:", r["choices"][0]["message"]["content"][:50])

    req2 = request.Request(base + "/v1/select",
                           data=json.dumps({"query": "q0", "n_blocks": 32,
                                            "top_k": 8}).encode(),
                           headers={"Content-Type": "application/json"})
    r2 = json.loads(request.urlopen(req2, timeout=10).read().decode())
    print("select:", r2)


if __name__ == "__main__":
    main()
