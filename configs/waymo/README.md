# Configs

### Common settings and notes

- The experiments are run with PyTorch 1.9 and CUDA 11.1.
- The training is conducted on 8 A100 GPUs. 
- Training on GPU with less memory would likely cause GPU out-of-memory. In this case, you can try configs with smaller batch size or frames.


### Waymo Validation Results

We provide the training and validation configs for the model in our paper. Let us know if you have trouble reproducing the results.

|         |  #Frame | Veh_L2 | Ped_L2 | Cyc_L2  | Mean   |
|---------|---------|--------|--------|---------|---------|
| [CenterFormer](voxelnet/waymo_centerformer.py)| 1       |   69.4     |  67.7      |   70.2      |  69.1    |
| [CenterFormer deformable](voxelnet/waymo_centerformer_deformable.py)| 1       |   69.7     |  68.3      |   68.8      |  69.0    |
| [CenterFormer](voxelnet/waymo_centerformer_multiframe_2frames.py)| 2       |   71.7     |  73.0      |   72.7      |  72.5    |
| [CenterFormer deformable](voxelnet/waymo_centerformer_multiframe_deformable_2frames.py)| 2       |   71.6     |  73.4      |   73.3      |  72.8    |
| [CenterFormer deformable](voxelnet/waymo_centerformer_multiframe_deformable_4frames.py)| 4       |   72.9     |  74.2      |   72.6      |  73.2    |
| [CenterFormer deformable](voxelnet/waymo_centerformer_multiframe_deformable_8frames.py)| 8       |   73.8     |  75.0      |   72.3      |  73.7    |
| [CenterFormer deformable](voxelnet/waymo_centerformer_multiframe_deformable_16frames.py)| 16      |   74.6     |  75.6      |   72.7      |  74.3    |