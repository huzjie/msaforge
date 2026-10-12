# 贡献指南

## 本地开发

```bash
python -m compileall -q msaforge
python -m unittest discover -s tests -t .
python -m msaforge doctor
```

## 约定

- 核心零依赖（只用标准库）。
- 确定性：所有随机都用 `utils/stable.py` 的 md5 种子函数。
- 训练信号必须单调可验证（skill 恒正 delta + 真正参与打分）。
- 每个新后端用 `@register_backend("name")` 并在 `backends/__init__.py` 显式 import。
