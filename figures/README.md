# 示例图说明

本目录只保留真实项目截图和样例，不再保留概念 SVG 图。样例用于说明 LiDAR 降面后的几何问题，以及语义分割如何为建筑重构提供部件级约束。

## 真实项目样例

`real/` 中的文件来自已有 HKUST(GZ) V6-r2、OpenMesh 和 V54 审阅产物，保留版本标识，便于组会直接查看：

- `v6r2_ground_material_review_preview.jpg`：真实地面材质分层和建筑排除结果（GitHub 预览版）。
- `v6r2_building_mask_alignment.png`：真实建筑排除掩膜。
- `lidar_direct_regularization_planar5deg.png`：真实 LiDAR 降面/规整审阅图，展示碎面、凹凸和细部混杂。
- `v54_component_95361_uv_montage.jpg`：真实组件纹理蒙太奇。

这些图是已有版本的审阅证据，不是同一输入下严格控制变量的算法 A/B 实验；报告中必须保留来源版本。
