import numpy as np

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

@PIPELINES.register_module
class Preprocess_multiframe(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        self.combine_frame = cfg.get("combine_frame", False)
        
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_noise = cfg.get('global_translate_noise', 0)
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None 
                
            self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        elif res["type"] in ["WaymoDataset_multi_frame"]:
            if self.combine_frame:
                points = res["lidar"]["combined"][0]
                previous_frame = res["lidar"]["combined"][1:]
                time_frame = [time[0,0] for time in res["lidar"]["times"]]
            else:
                points = res["lidar"]["points"][0]
                previous_frame = res["lidar"]["points"][1:]
                time_frame = [time[0,0] for time in res["lidar"]["times"]]
        elif res["type"] in ["NuScenesDataset"]:
            points = res["lidar"]["combined"]
        else:
            raise NotImplementedError

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }

        points_num = []
        points_timeframe = []

        if self.mode == "train" and not self.no_augmentation:
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    # res["metadata"]["num_point_features"],
                    points.shape[1],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )


                    points = np.concatenate([sampled_points, points], axis=0)
                    points_num.append(points.shape[0])
                    points_timeframe.append(0.)

                    if res["type"] in ["WaymoDataset_multi_frame"]:
                        for idx, pre_points in enumerate(previous_frame):
                            pre_points = np.concatenate([sampled_points, pre_points], axis=0)
                            points_num.append(pre_points.shape[0])
                            points = np.concatenate([points, pre_points], axis=0)
                            points_timeframe.append(time_frame[idx+1])
                else:
                    points_num.append(points.shape[0])
                    points_timeframe.append(0.)
                    if res["type"] in ["WaymoDataset_multi_frame"]:
                        for idx, pre_points in enumerate(previous_frame):
                            points_num.append(pre_points.shape[0])
                            points = np.concatenate([points, pre_points], axis=0)
                            points_timeframe.append(time_frame[idx+1])
            else:
                points_num.append(points.shape[0])
                points_timeframe.append(0.)
                if res["type"] in ["WaymoDataset_multi_frame"]:
                    for idx, pre_points in enumerate(previous_frame):
                        points_num.append(pre_points.shape[0])
                        points = np.concatenate([points, pre_points], axis=0)
                        points_timeframe.append(time_frame[idx+1])

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
            
            gt_dict["gt_boxes"], points = prep.global_rotation(
                gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
            )
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise
            )
            gt_dict["gt_boxes"], points = prep.global_translate_v2(
                gt_dict["gt_boxes"], points, noise_translate=self.global_translate_noise
            )
        elif self.no_augmentation:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            points_num.append(points.shape[0])
            points_timeframe.append(0.)
            if res["type"] in ["WaymoDataset_multi_frame"]:
                for idx, pre_points in enumerate(previous_frame):
                    points_num.append(pre_points.shape[0])
                    points = np.concatenate([points, pre_points], axis=0)
                    points_timeframe.append(time_frame[idx+1])
        else:
            points_num.append(points.shape[0])
            points_timeframe.append(0.)
            if res["type"] in ["WaymoDataset_multi_frame"]:
                for idx, pre_points in enumerate(previous_frame):
                    points_num.append(pre_points.shape[0])
                    points = np.concatenate([points, pre_points], axis=0)
                    points_timeframe.append(time_frame[idx+1])

        #disengage points in multi-frame
        if len(points_num)>1:
            previous_frames = []
            counts = 0
            for n in range(len(points_num)):
                previous_frames.append(points[counts:counts+points_num[n]])
                if self.shuffle_points:
                    np.random.shuffle(previous_frames[-1])
                counts += points_num[n]
            # points = points[:points_num[0]]
            res["lidar"]["multi_points"] = previous_frames
        else:
            if self.shuffle_points:
                np.random.shuffle(points)
            res["lidar"]["points"] = points

        res["lidar"]["times"] = np.asarray(points_timeframe)

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info