# 示例图说明

本目录的 SVG 是用于解释方法动机的脱敏概念图，不代表真实校园数据的定量结果。真实运行时，应在相同坐标、相同输入和相同颜色图例下替换或补充实测对比图。

| 文件 | 解释的问题 | 关键结论 |
|---|---|---|
| `01_fragmentation_regularization.svg` | LiDAR 三角面碎化与直接简化 | 先识别共面连通域，再规整边界，避免只减面数造成锯齿和形变 |
| `02_registration_layers.svg` | 建筑、地面和 LiDAR 坐标错位 | 先统一 ENU 和镜像合同，再做建筑排除和地面贴合 |
| `03_component_material_consistency.svg` | 同一玻璃/墙面被逐三角面分成多种材质 | 以连通域为单位融合语义，再传播到三角面 |
| `04_evidence_to_asset.svg` | 从多源证据到仿真资产 | 几何、语义、材质和审计分层，避免纹理被误当电磁真值 |

图中“效果”表示方法目标或可检验属性，不表示已经完成的实测提升。

## 真实项目样例

`real/` 中的文件来自已有 HKUST(GZ) V6-r2、OpenMesh 和 V54 审阅产物，保留版本标识，便于组会直接查看：

- `v6r2_ground_material_review_preview.jpg`：真实地面材质分层和建筑排除结果（GitHub 预览版）。
- `v6r2_building_mask_alignment.png`：真实建筑排除掩膜。
- `lidar_direct_regularization_planar5deg.png`：真实 LiDAR 规整审阅图。
- `v54_component_95361_uv_montage.jpg`：真实组件纹理蒙太奇。

这些图是已有版本的审阅证据，不是同一输入下严格控制变量的算法 A/B 实验；报告中必须保留来源版本。
