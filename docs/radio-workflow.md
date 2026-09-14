# 无线预测交付与复现

## 运行环境

Python 3.10+。安装 `.[runtime,radio]` 可运行全部十专家的 CPU 拟合。Torch 用于精确 GP 优化，名称为 radio-neural 的依赖组并不表示本报告使用 U-Net。RT 单独使用 `.[rt]`，对应 Sionna RT 2.0.1 / Mitsuba 3.8.0。GPU/驱动需在部署机器验证。不要将不同版本计算的 RT 缓存混合使用。

## 通用数据合同

一个目录包含以下三个文件：

| 文件 | 必需字段/形状 |
| --- | --- |
| points.csv | point_id, x, y, z, band, role, observed_dbm |
| rt.npz | point_id[N], gains[C,N], los[N], tx[3] |
| configs.json | C 个配置字典，首项 id=AUTO_BASE, family=BASE, groups={} |

`role` 为 train/query。训练行有 dBm 标签，查询行的 observed_dbm 必须为空。至少 24 个训练点及一个查询点，并且内层三折空间划分可行。缺失 RT 路径用 NaN，不能用零冒充缺失。NPZ 使用普通数值/字符串数组，不能依赖 pickle。

点、路径、建筑和发射机必须处于同一个局部米制坐标系。不要将经纬度直接作为 x/y。固定坐标原点、轴方向、旋转和高程基准，转换过程写入数据侧的配准记录。接收机高度使用实际实验配置。RT 的 point_id 顺序必须与 CSV 一致，入口不按行数猜测对应关系。训练/查询不能共享 1 m 位置分组。

所有 RT 配置的 gains 必须已展开到每个接收点。当前 aligned 校正使用首个 AUTO_BASE prior，根据训练支持与 LOS 条件拟合功率映射。它不是任意接收功率 CSV 的自动单位转换器。

## 单批次预测

```text
python tools/run_radio.py inspect --input data/n41
python tools/run_radio.py predict --input data/n41 --output runs/n41 --mode infill --device cpu
python tools/run_radio.py score --predictions runs/n41/predictions.csv --truth data/n41-truth.csv --output runs/n41/metrics.csv
```

`spatial30` 和 `spatial60` 分别使用 30/60 m 内层空间排除距离，不可行时停止而不是更换协议。该参数控制内层验证，外部测试集如何留出由数据制作者单独执行。在 `predict` 中不加入 `--quick` 才是正常 GP 拟合。输出目录须为空，避免覆盖前次结果。

入口先用训练和查询坐标从八组候选划分中选取空间支持最接近的一组。每个验证折独立拟合训练功率校正和十专家，仅在验证标签上比较误差。最终使用全部训练点重拟合，再预测无标签查询点。SELECT_MAE/SELECT_RMSE 是整批查询选择一位专家，LOCAL_RISK 输出依据邻域验证误差加权组合。查询真实信号只在独立 `score` 命令中读取。

该入口是便于交付的新适配层，复用原算法，不等同于冻结的 240 任务报告协议。原协议的种子、划分、运行哈希和评分合同由下面的 legacy 入口管理。

## 完整冻结实验

H18 保留 n41/n79、infill/spatial30/spatial60、30/100/300/1000/2500 点预算、外层 fold3/fold4、四次重复，以及十专家加八种选择输出。全部条件可行时合计 240 任务。材质对照使用 semantic_8class 与 uniform_nonconductor_concrete，共 480 次拟合任务。

必须提供原协议文件、经过合同校验的旧 RT 缓存与点数据，或完整的已准备 fold3/fold4 目录。摘要 CSV 不能代替准备数据。

```text
python -m src.radio.legacy.run_h18_distance_trend prepare --data-root data/measurements --rt-root data/corrected-rt --output-dir runs/h18 --protocol config/my-frozen-protocol.json
python -m src.radio.legacy.run_h18_distance_trend run --root runs/h18 --key RUN_ID --device cpu
python -m src.radio.legacy.run_h18_distance_trend score --root runs/h18
```

`RUN_ID` 必须取自生成的 manifest，不手工发明。评分会等待完整冻结结果，检查划分与输入哈希。准备阶段读取 data-root/cache/BAND 下的 points.csv、point_to_unique.npy 和 unique_xyz.npy。corrected RT 目录包含受 manifest 绑定的 configs.json、路径增益与 LOS 缓存，其中旧搜索合同需要 13 个配置，而不是通用入口的简化 AUTO_BASE 格式。完整字段由 `load_corrected_h13_band` 校验。

在此基础上重算配准与材质对照：

```text
python -m src.radio.legacy.aligned_factorial prepare --job-root runs/aligned --results-root runs/h18 --transform-json data/alignment.json --scene-xml data/scene.xml
python -m src.radio.legacy.aligned_factorial smoke-trace --job-root runs/aligned --scene-xml data/scene.xml
python -m src.radio.legacy.aligned_factorial trace --job-root runs/aligned --scene-xml data/scene.xml --arm semantic_8class --band n41
python -m src.radio.legacy.aligned_factorial run-case --job-root runs/aligned --results-root runs/h18 --key RUN_ID --device cpu
python -m src.radio.legacy.aligned_factorial score --job-root runs/aligned --results-root runs/h18
```

trace 需遍历两个 band × 两个 arm，再运行 manifest 中全部 key。运行会按哈希检查分块缓存和 receiver geometry，改变配准或场景后不能复用旧缓存。controller 的 `--help` 提供自动调度入口及独立 RT/拟合 Python 环境设置。

## 场景、测量和版本交接

建模入口生成 OBJ/MTL 和几何/材质记录。RT 入口消费 Sionna 可加载的 XML 与所引用网格。普通视觉 MTL 不能直接当作电磁材质配置，场景导出及材质族对应应在 Blender/Sionna 环境中明确检查。本源码包不含私有 XML/网格，未把缺失的场景转换伪装为已实现步骤。

采集原始记录需在数据准备阶段统一位置与信号字段，并保存频段、时间、批次和路径对应。项目使用推车承载 DJI M350 RTK 定位设备与衡龙信号测试仪同时采集。具体原始 CSV 列名、天线高度和部署坐标由现场数据记录确定，不在通用接口中猜测。通用入口采用标准化 CSV，不声称直接解码所有仪器格式。

溯源文件记录每个迁入脚本的原始 SHA-256，移植后文件的 SHA-256 由发布清单提供。旧实验产物绑定旧运行哈希，不应为兼容新代码手改原 manifest。完整重跑使用新的准备目录。

## 交付验证

`python -m pytest -q` 检查建模契约、标签隔离、ID 对齐、选择和发布规则。`python tools/run_radio.py demo --output runs/check` 使用合成路径增益调用真实十专家及 MATCHED 验证。它验证软件链路，不是 RT 精度实验。

真实场景 RT、原始采集解析、完整 240 任务重跑还需要私有资产与对应运行环境。发布包只包含可审查源码、配置示例、文档和测试。原始数据、运行缓存和密钥不进入 Git。
