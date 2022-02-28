from os import listdir
from os.path import join
import os
import numpy as np
from datasets.base_dataset import BaseDataset
from utils.util import resample_pcd
import open3d as o3d
from utils.plyfile import load_ply
from datasets.shapenet_3depn import sample_point_cloud_by_n

class HeapDataset(BaseDataset):
    def __init__(self, root_dir, mode="train"):
        super().__init__(root_dir)
        self.mode = mode
        self.completion_path = join(root_dir, mode, 'complete')
        self.existing_path = join(root_dir, mode, 'existing')
        self.missing_path = join(root_dir, mode, 'missing')
        self.file_names = os.listdir(self.completion_path)
        
    def _get_scales(self, pcd):
        axis_mins = np.min(pcd.T, axis=1)
        axis_maxs = np.max(pcd.T, axis=1)
        scale = np.max(axis_maxs - axis_mins)
        pcd_center = (axis_maxs + axis_mins) / 2
        return pcd_center, scale / 0.9

    def perform_scaling(self, pcd, pcd_center, scale):
        pcd_center, scale = self._get_scales(pcd)
        pcd = (pcd - pcd_center) / scale
        return pcd

    def load_ply(self, path):
        pcd = o3d.io.read_point_cloud(path)
        pcd = np.asarray(pcd.points).astype(np.float32)
        return pcd

    def __getitem__(self, idx):
        existing = load_ply(join(self.existing_path, self.file_names[idx]))
        missing = load_ply(join(self.missing_path, self.file_names[idx]))
        gt = load_ply(join(self.completion_path, self.file_names[idx]))
        pcds = [existing, missing, gt]
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd_center, scale = self._get_scales(pcd)
            pcd = self.perform_scaling(pcd, pcd_center, scale)
            pcd = sample_point_cloud_by_n(pcd, 2048)
            pcds[i] = pcd
        return pcds[0],pcds[1],pcds[2], idx

    def inverse_scale(self, idx, scaled_pcd):
        pcd = self.load_ply(self.data_paths[idx])
        pcd_center, scale = self._get_scales(pcd)
        scaled_pcd_center, scaled_pcd_scale = self._get_scales(scaled_pcd)
        return (scaled_pcd / scaled_pcd_scale * scale) + pcd_center

    def __len__(self):
        return len(self.completion_path)

    @classmethod
    def get_validation_datasets(cls, root_dir, classes=[], **kwargs):
        raise NotImplementedError
    
    @classmethod
    def get_test_datasets(cls, root_dir, classes=[], **kwargs):
        return {'all': HeapDataset(root_dir=root_dir)}
