# 快速上手

## 安装（零依赖，直接跑）

```bash
git clone https://github.com/huzjie/msaforge
cd msaforge
python -m msaforge doctor
```

## 五个子命令

| 命令 | 作用 |
|---|---|
| `python -m msaforge doctor` | 校验五后端 + 组件 |
| `python -m msaforge train` | 训练 index 分支（skill → 1.0） |
| `python -m msaforge bench` | 跑 speed/recall/quality 基准 |
| `python -m msaforge select "query"` | 展示稀疏选块 |
| `python -m msaforge serve` | 启动 OpenAI 兼容 HTTP 服务 |

## HTTP 服务

```bash
python -m msaforge serve --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

## Docker

```bash
docker build -t msaforge .
docker run -p 8000:8000 msaforge
```
