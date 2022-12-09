# CenterFormer
Official implementation for [**CenterFormer: Center-based Transformer for 3D Object Detection**](https://arxiv.org/abs/2209.05588) (ECCV 2022 Oral)
```
@InProceedings{Zhou_centerformer,
title = {CenterFormer: Center-based Transformer for 3D Object Detection},
author = {Zhou, Zixiang and Zhao, Xiangchen and Wang, Yu and Wang, Panqu and Foroosh, Hassan},
booktitle = {ECCV},
year = {2022}
}
```

## Highlights
- **Center Transformer** We introduce a center-based transformer network for 3D object detection. 

- **Fast and Easy to Train** We use the center feature as the initial query embedding to facilitate learning of the transformer. We propose a multi-scale cross-attention layer to efficiently aggregate neighboring features without significantly increasing the computational complexity.

- **Temporal information**: We propose using the cross-attention transformer to fuse object features from past frames.

<p align="center"> <img src='docs/mtf_architecture_eccv.png' align="center" height="500px"> </p>

## NEWS
[2022-12-09] Add support for multi-task head and nuScenes training configs. 

[2022-09-30] CenterFormer source code is released. 

## Abstract
Query-based transformer has shown great potential in constructing long-range attention in many image-domain tasks, but has rarely been considered in LiDAR-based 3D object detection due to the overwhelming size of the point cloud data. In this paper, we propose **CenterFormer**, a center-based transformer network for 3D object detection. CenterFormer first uses a center heatmap to select center candidates on top of a standard voxel-based point cloud encoder. It then uses the feature of the center candidate as the query embedding in the transformer. To further aggregate features from multiple frames, we design an approach to fuse features through cross-attention. Lastly, regression heads are added to predict the bounding box on the output center feature representation. Our design reduces the convergence difficulty and computational complexity of the transformer structure. The results show significant improvements over the strong baseline of anchor-free object detection networks. CenterFormer achieves state-of-the-art performance for a single model on the Waymo Open Dataset, with 73.7% mAPH on the validation set and 75.6% mAPH on the test set, significantly outperforming all previously published CNN and transformer-based methods.

## Result

#### 3D detection on Waymo test set 

|         |  #Frame | Veh_L2 | Ped_L2 | Cyc_L2  | Mean   |
|---------|---------|--------|--------|---------|---------|
| CenterFormer| 8       |   77.7     |  76.6      |   72.4      |  75.6    |
| CenterFormer| 16      |   78.3     |  77.4      |   73.2      |  76.3    |

#### 3D detection on Waymo val set 

|         |  #Frame | Veh_L2 | Ped_L2 | Cyc_L2  | Mean   |
|---------|---------|--------|--------|---------|---------|
| [CenterFormer](configs/waymo/voxelnet/waymo_centerformer.py)| 1       |   69.4     |  67.7      |   70.2      |  69.1    |
| [CenterFormer deformable](configs/waymo/voxelnet/waymo_centerformer_deformable.py)| 1       |   69.7     |  68.3      |   68.8      |  69.0    |
| [CenterFormer](configs/waymo/voxelnet/waymo_centerformer_multiframe_2frames.py)| 2       |   71.7     |  73.0      |   72.7      |  72.5    |
| [CenterFormer deformable](configs/waymo/voxelnet/waymo_centerformer_multiframe_deformable_2frames.py)| 2       |   71.6     |  73.4      |   73.3      |  72.8    |
| [CenterFormer deformable](configs/waymo/voxelnet/waymo_centerformer_multiframe_deformable_4frames.py)| 4       |   72.9     |  74.2      |   72.6      |  73.2    |
| [CenterFormer deformable](configs/waymo/voxelnet/waymo_centerformer_multiframe_deformable_8frames.py)| 8       |   73.8     |  75.0      |   72.3      |  73.7    |
| [CenterFormer deformable](configs/waymo/voxelnet/waymo_centerformer_multiframe_deformable_16frames.py)| 16      |   74.6     |  75.6      |   72.7      |  74.3    |

#### 3D detection on nuScenes val set
|         |  NDS    | mAP    |
|---------|---------|--------|
| [CenterFormer](configs/nusc/nuscenes_centerformer_separate_detection_head.py)| 68.0     |  62.7      |
| [CenterFormer deformable](configs/nusc/nuscenes_centerformer_deformable_separate_detection_head.py)| 68.4     |  63.0      |

The training and evaluation configs of the above models are provided in [Configs](configs/waymo/README.md).

## Installation
Please refer to [INSTALL](docs/INSTALL.md) to set up libraries needed for distributed training and sparse convolution.

## Training and Evaluation
Please refer to [WAYMO](docs/WAYMO.md) and [nuScenes](docs/NUSC.md) to prepare the data, training and evaluation.


## Acknowlegement
This project is developed based on the [CenterPoint](https://github.com/tianweiy/CenterPoint) codebase. We use the deformable cross-attention implementation from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR).
