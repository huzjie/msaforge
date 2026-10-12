# 配方 04：搭一个 OpenAI 兼容的稀疏推理服务

## 步骤

1. **启动服务**：
   ```bash
   python -m msaforge serve --port 8000
   ```
2. **健康检查**：`curl http://127.0.0.1:8000/health`
3. **补全**：
   ```bash
   curl -X POST http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"hello"}]}'
   ```
4. **选块**：
   ```bash
   curl -X POST http://127.0.0.1:8000/v1/select \
     -H "Content-Type: application/json" \
     -d '{"query":"q0","n_blocks":32,"top_k":8}'
   ```

## 对接
`/v1/chat/completions` 是标准 OpenAI 接口，可直接把 `base_url` 指到 `http://127.0.0.1:8000`
接入 Cursor / Claude Code 等工具（参考实现用 mock 后端，替换成真实后端即生产）。
