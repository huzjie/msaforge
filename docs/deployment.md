# 部署指南

## Docker

```bash
docker build -t msaforge .
docker run -p 8000:8000 msaforge
```

## Kubernetes

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

## Helm

```bash
helm install msaforge deploy/helm/msaforge
```

## 作为 OpenAI 兼容端点接入

把客户端 `base_url` 设为 `http://<host>:8000`（参考实现用 mock 后端，替换为
openai/vllm/transformers 后端即接真实模型）。
