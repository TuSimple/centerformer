## Getting Started with CenterFormer on nuScenes
Modified from [CenterPoint](https://github.com/tianweiy/CenterPoint)

### Prepare data

#### Download data and organise as follows

```
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata
```

Create a symlink to the dataset root 
```bash
mkdir data && cd data
ln -s DATA_ROOT 
mv DATA_ROOT nuscenes
```
Remember to change the DATA_ROOT to the actual path in your system. 


#### Create data

Data creation should be under the gpu environment.

```
# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10
```

In the end, the data and info files should be organized as follows

```
# For nuScenes Dataset 
└── centerformer
       └── data    
              └── nuscenes 
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
                     |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations
                     |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo <-- GT database 
```

### Train & Evaluate

```bash
python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py CONFIG_PATH
```

```bash
python -m torch.distributed.launch --nproc_per_node=8 ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth 
```
