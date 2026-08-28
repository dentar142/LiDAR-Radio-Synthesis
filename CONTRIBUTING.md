# Contributing

本仓库当前以方法骨架和接口契约为主。贡献代码时请保持阶段边界，不要把原始校园模型、私有坐标控制点、密码、令牌或未公开数据提交到仓库。

提交前至少运行：

```text
python -m compileall -q src tests tools
python tools/run_stage.py --config config/example_scene.yaml --stage all
```

实现真实算法时，应先为新增行为添加脱敏小样例测试，并在报告中说明输入来源、参数、坐标基准和物理参数来源。
