# Configs

### Common settings and notes

- The experiments are run with PyTorch 1.9 and CUDA 11.1.
- The training is conducted on 8 A100 GPUs. 
- Training on GPU with less memory would likely cause GPU out-of-memory. In this case, you can try configs with smaller batch size or frames.


### nuScenes Validation Results

|         |  NDS    | mAP    |
|---------|---------|--------|
| [CenterFormer](nuscenes_centerformer_separate_detection_head.py)| 68.0     |  62.7      |
| [CenterFormer deformable](nuscenes_centerformer_deformable_separate_detection_head.py)| 68.4     |  63.0      |


