# Data sources and availability

## Three-dimensional environment asset

Repository: [LITIANSHUN/HKUST_GZ_3Dcampus](https://github.com/LITIANSHUN/HKUST_GZ_3Dcampus).

The project owner identified this repository as the source of the original campus model on 2026-09-14. Its README describes UAV oblique photogrammetry, including reconstruction and texture mapping, with OBJ, PLY, OSGB and B3DM outputs. Accordingly, describe this source as a photogrammetric textured 3D model rather than a LiDAR-acquired dataset unless separate acquisition evidence establishes a LiDAR input.

The repository recommends citing Tianshun Li, Tianyi Huai, Zhen Li, Yichun Gao, Haoang Li and Xinhu Zheng, “SkyVLN: Vision-and-Language Navigation and NMPC Control for UAVs in Urban Environments,” 2025, arXiv:2507.06564. The linked arXiv record notes acceptance at IROS 2025:
https://arxiv.org/abs/2507.06564

The repository URL identifies the asset source. The associated paper identifies the source project, not the radio-map prediction method implemented here. Retain the source repository's usage conditions when obtaining assets. The model is not bundled in this source-code release.

## Wireless measurements

The project team's ground measurements are publicly available at https://huggingface.co/datasets/Neko142/pi-razer-ground-measurements. The verified upload revision is 28b42377975c44f14ec7d8039fd6c17ade3dfe9d (2026-09-14). It contains 20 byte-preserving gzip-compressed instrument CSV exports plus metadata and a dataset card. All 26 remote files, including the Hub-generated .gitattributes, passed checksum verification. These raw exports support the n41/n79 analysis but are not a prefiltered experiment split.

Public visibility was verified on 2026-09-14. A redistribution license is not asserted here; consult the dataset card for current terms. Do not substitute the 3D model repository for the wireless-measurement citation.

## Processing code

Pipeline repository: https://github.com/dentar142/LiDAR-Radio-Synthesis

The three citations serve distinct purposes: environment assets, measured radio data and processing software.
