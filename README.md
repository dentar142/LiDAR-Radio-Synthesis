# How to Make a Clean Map for Simulation

## π-Razer：建模与无线预测代码交付

仓库现在同时包含前半段几何/材质重建和后半段无线预测代码。后半段接入实际 H18 十专家、空间匹配验证、校正功率和双材质对照实验控制器，数值核心保留在 `src/radio/legacy/`，移植来源及原始摘要见 `docs/radio-source-provenance.json`。

流程为：工程图与 LiDAR → 几何及表面材质 → RT 场景与路径增益 → 训练点功率校正 → 十专家拟合 → 查询空间匹配验证 → 专家选择/风险加权 → 独立测试评估。

### 快速运行

```text
python -m pip install -e ".[runtime,radio,test]"
python -m pytest -q
python tools/run_radio.py demo --output runs/radio-smoke
```

演示使用固定种子的合成数据，实际调用十专家与三折选择代码，不需要校园资产或 GPU。输出包括逐点预测、验证预测、权重、选择审计和软件测试指标。演示的快速 GP 拟合与合成指标不用于报告实验结论。

真实数据入口：

```text
python tools/run_radio.py inspect --input data/my-band
python tools/run_radio.py predict --input data/my-band --output runs/my-band
python tools/run_radio.py score --predictions runs/my-band/predictions.csv --truth data/private-truth.csv --output runs/my-band/metrics.csv
```

每个输入目录只放一个频段，查询标签不进入预测文件。格式、坐标合同、RT 前置条件和完整实验命令见 [无线流程与复现](docs/radio-workflow.md)。

### 目录与交付边界

| 目录/入口 | 内容 |
| --- | --- |
| `src/reconstruction/` | 工程图/LiDAR 配准、实体重建、材质映射 |
| `src/radio/` | 数据检查、预测入口、合成测试数据 |
| `src/radio/legacy/` | 实验数值核心、RT、冻结划分、双材质比较 |
| `tools/run_stage.py` | 原有建模入口 |
| `tools/run_radio.py` | inspect / predict / score / demo |
| `tools/build_release.py` | 源码包及逐文件 SHA-256 清单 |

原始测量数据、模型、RT 缓存与接收机真值由使用者提供，不随源码发布。通用入口覆盖单个查询批次的 MATCHED 选择及风险加权，冻结实验入口覆盖 FIXED/MATCHED、双外层划分与材质消融。两种入口不混用实验清单。

原始三维环境模型来源为 [HKUST_GZ_3Dcampus](https://github.com/LITIANSHUN/HKUST_GZ_3Dcampus)，该仓库描述的是无人机倾斜摄影测量重建。无线测量数据已公开发布于 [Hugging Face](https://huggingface.co/datasets/Neko142/pi-razer-ground-measurements)。来源与推荐引用见 [数据来源说明](docs/data-sources.md)。

打包命令：`python tools/build_release.py --output-dir release`。本地打包不会提交或推送 GitHub。

以下保留原有建模说明，其中模型统计对应各自的建模版本，不应与后续报告版本合并统计。

本仓库提供一条可执行的全自动基线：从建筑工程图/总平图提取规则轮廓，以原始 LiDAR/摄影测量 OBJ 的高度证据完成跨模态配准、局部地面估计和多屋面层次拆分，生成封闭几何白模；随后把原始纹理网格的面级视觉语义通过三维体素回投影到新模型，并在连通表面尺度聚合材质候选。

## 范围

- 输入：原始纹理 LiDAR/摄影测量 OBJ、工程图或总平图、可选的原网格语义体素索引。
- 输出：规则化封闭 OBJ/MTL、建筑高度与部件记录、连通表面材质候选、配准叠加图和机器审计清单。
- 自动化范围：无需人工面片标注即可执行完整流水；低覆盖或多源冲突不会被强行当作真值，而会保留为 `REVIEW_REQUIRED`。
- 科学边界：视觉语义标签不等于施工材料真值，更不等于介电常数、电导率或频率相关电磁参数。

## 当前状态

`src/reconstruction/` 是可运行的自动基线，包含流式 OBJ 栅格化、工程图轮廓提取、跨模态配准、LiDAR 高度拟合、封闭实体构建和原模型材质回投影。其他 `src/` 模块保留通用阶段契约。当前实现是研究基线，并非测绘级或竣工 BIM 重建工具。

## 建议入口

先安装运行依赖：

```text
python -m pip install -e ".[runtime]"
```

```text
python tools/run_stage.py \
  --config config/hkustgz_auto_reconstruction.example.yaml \
  --stage all
```

真实运行前应把配置中的绝对路径替换为服务器上的只读源数据路径。大 OBJ 只做流式扫描，生成物写入独立 `runs/` 目录。

## 自动流水

```text
工程图建筑填色 -> 规则 footprint -> LiDAR 正射高度证据 -> 自动跨模态配准
-> 局部地面/屋顶高度 -> 多高度连通域 -> 封闭规则几何
-> 原始纹理网格语义体素回投影 -> 连通表面投票与置信度门禁
-> OBJ/MTL + 审计 JSON
```

材质转移采用“表面连通域”而不是“每个三角面独立分类”。同一玻璃幕墙或共面墙体会先汇总原模型证据，再统一给出标签、命中率、纯度、置信度与审阅状态，从结构上减少碎三角面导致的材质跳变。

## 远端实跑证据

完整校园数据的无人工标注试跑已通过全部运行门禁：从总平图自动恢复 35 栋建筑轮廓，32 栋直接取得 LiDAR 高度，拆分为 160 个封闭建筑部件，输出 10,304 个三角形和 3,031 个连通表面；每个部件的非流形边数为 0。材质回投影平均原网格语义命中率为 0.5984，其中 1,377 个表面达到自动候选阈值，1,654 个表面保留为 `REVIEW_REQUIRED`。整链耗时约 54 秒。

这里的运行状态为 `operational_status=PASS`，科学状态仍为 `REVIEW_REQUIRED`。原因是当前没有独立 GCP、竣工 BIM、人工材质真值和现场电磁测量；自动标签只能作为后续人工审阅或材料参数标定的候选。

方法、实现细节和历史演进见：

- [`docs/automatic-plan-lidar-material-pipeline_zh.md`](docs/automatic-plan-lidar-material-pipeline_zh.md)：当前全自动几何与材质回投影流水。
- [`docs/complete-work-report_zh.md`](docs/complete-work-report_zh.md)：从早期 LiDAR 规整、视觉语义实验到 Sionna 部署的完整工作报告。

## 科研演示动画

[`film/`](film/) 提供 155 秒发布会风格科研动画的可复现工程，包含中文分镜、旁白、Blender 点云/模型动画、程序化音乐、字幕和服务器合成脚本。视频使用真实项目资产，但原始模型和渲染帧不进入 Git 仓库。

## 真实问题样例

见 [`docs/figure-guide.md`](docs/figure-guide.md)。文档只引用已有项目截图，重点解释 LiDAR 降面后的几何失真，以及语义分割如何按楼体、墙面、屋顶和玻璃等建筑特性组织重构。

## 自适应语义体系

这里的“自适应”不是更换一个固定分割模型，而是让系统根据区域尺度、几何连续性和证据可靠度动态选择处理粒度与重构策略：

```text
场景 -> 建筑 -> 部件 -> 表面连通域 -> 三角面
```

- 大面积、近共面的墙面或屋顶：合并连通域并执行平面规整。
- 玻璃幕墙：保持语义连续性，避免逐三角面材质跳变。
- 建筑边缘、窗框和突出构件：保护边界并保留细节。
- 纹理受阴影、反射或遮挡影响时：降低视觉证据权重，提高几何或人工证据权重。
- 多源证据冲突且置信度不足时：停止自动合并并进入人工复核。

因此，同一套系统不会对所有三角面使用相同阈值或相同简化策略，而是根据建筑特性选择 `regularize_plane`、`preserve_detail`、`merge_component`、`keep_boundary` 或 `manual_review`。
