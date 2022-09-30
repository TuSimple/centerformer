import torch
import numpy as np
import sys

def compare_models(model_1, model_2):
    models_differ = 0

    print(model_1["state_dict"]['neck.transformer_layer.layers.1.2.fn.net.0.weight'])

    print(model_2["state_dict"]['neck.transformer_layer.layers.1.2.fn.net.0.weight'])

    # for key_item_1, key_item_2 in zip(model_1["state_dict"].items(), model_2["state_dict"].items()):
    #     if torch.equal(key_item_1[1], key_item_2[1]):
    #         # print('match found at', key_item_1[0])
    #         pass
    #     else:
    #         models_differ += 1
    #         if (key_item_1[0] == key_item_2[0]):
    #             print('Mismtach found at', key_item_1[0])
    #             # print(key_item_1[1])
    #             # print(key_item_2[1])
    #             # raise Exception
    #         else:
    #             raise Exception
    # if models_differ == 0:
    #     print('Models match perfectly! :)')

model1_path='/mnt/truenas/scratch/zixiang.zhou/code/CenterPoint/work_dirs/waymo_centerpoint_voxelnet_transforemer_SCA_multiscale_CBAM_test_6epoch/pre_epoch_11.pth'
model2_path='/mnt/truenas/scratch/zixiang.zhou/code/CenterPoint/work_dirs/waymo_centerpoint_voxelnet_transforemer_SCA_multiscale_CBAM_test_6epoch/pre_epoch_19.pth'

checkpoint1 = torch.load(model1_path, map_location="cpu")
checkpoint2 = torch.load(model2_path, map_location="cpu")

compare_models(checkpoint1,checkpoint2)
