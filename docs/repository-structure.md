# 仓库结构

```text
HKUSTGZ-material-mapping/
├── config/       场景配置和坐标合同示例
├── docs/         方法论、数据边界和复现说明
├── src/          七个阶段的类型、接口和后续算法实现
├── tests/        脱敏样例与契约测试
├── tools/        阶段化 CLI 和检查工具
├── data/         本地输入目录，不进入 Git
├── runs/         运行输出目录，不进入 Git
└── archives/     日期归档目录，不进入 Git
```

推荐执行顺序为 `geometry -> building -> semantic -> materials -> ground -> georef -> export`。每个阶段应输出可追溯的中间对象、来源信息和审计状态，失败时阻断下游阶段。
