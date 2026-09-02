# 科研演示动画工程

本目录把自动重构流水制作成约 155 秒的中文科研动画。画面使用真实 LiDAR/摄影测量模型、总平图、配准叠加图、自动清洁模型和机器审计数字；中间帧与最终视频在服务器生成，不把原始校园数据复制进 Git 仓库。

## 文件

- `storyboard_zh.md`：分镜、节奏和色彩规范。
- `narration_zh.txt`：中文旁白原稿。
- `prepare_film_assets.py`：从只读源图生成 1080p 展示纹理。
- `render_film.py`：Blender 4.5 场景、相机、点云、建筑生长、材质扫描和数据卡片动画。
- `make_audio_bed.py`：无版权依赖的程序化环境音乐。
- `assemble_film.py`：神经语音、字幕、音乐与 1080p 画面的最终合成。

## 渲染阶段

```text
prepare_film_assets.py
        ↓
render_film.py --preview --stills   # 关键帧视觉审阅
        ↓
render_film.py --preview            # 低清全片节奏审阅
        ↓
render_film.py                      # 1080p 正片
        ↓
assemble_film.py                    # 旁白、字幕、音乐和 MP4
```

Blender 脚本默认生成 1920×1080、30 fps、4650 帧的视频。`--preview` 改为 960×540；`--stills` 只渲染九个代表帧。源 OBJ 保持只读，点云采用确定性 reservoir sampling，最大保留 95,000 个可视点，不生成大型本地副本。

旁白使用约 149.5 秒的中文神经语音，成片末尾保留约 5 秒音乐收束。字幕采用 Noto Sans CJK，并由 ASS 渲染以保证 Linux 服务器上的中文字形一致。

## 科学表达边界

视频中的 `operational PASS` 只表示轮廓、配准、高度、几何和材质回投影门禁通过。材质颜色是视觉/几何候选，不是施工记录，也不是介电常数、电导率或频率相关电磁参数真值；因此片中同时保留 `scientific REVIEW_REQUIRED`。
