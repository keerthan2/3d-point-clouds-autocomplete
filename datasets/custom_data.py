from os import listdir
from os.path import join
import os
import numpy as np
from datasets.base_dataset import BaseDataset
from utils.util import resample_pcd
import open3d as o3d

class CustomDataset(BaseDataset):

    def __init__(self, root_dir, mode="test"):
        super().__init__(root_dir)
        self.mode = mode
        self.data_paths = []
        for f in listdir(self.root_dir):
            self.data_paths.append(os.path.join(root_dir,f))
        
    def _get_scales(self, pcd):
        axis_mins = np.min(pcd.T, axis=1)
        axis_maxs = np.max(pcd.T, axis=1)

        scale = np.max(axis_maxs - axis_mins)
        pcd_center = (axis_maxs + axis_mins) / 2

        return pcd_center, scale / 0.9

    def load_ply(self, path):
        pcd = o3d.io.read_point_cloud(path)
        pcd = np.asarray(pcd.points).astype(np.float32)
        return pcd

    def __getitem__(self, idx):
        pcd = self.load_ply(self.data_paths[idx])
        pcd_center, scale = self._get_scales(pcd)
        pcd = (pcd - pcd_center) / scale
        # print(pcd.shape, np.min(pcd,axis=0),np.max(pcd,axis=0))
        # exit()
        # return resample_pcd(pcd, 2048), 0, 0, idx
        return resample_pcd(pcd, 10000), 0, 0, idx

    def inverse_scale(self, idx, scaled_pcd):
        pcd = self.load_ply(self.data_paths[idx])
        pcd_center, scale = self._get_scales(pcd)
        scaled_pcd_center, scaled_pcd_scale = self._get_scales(scaled_pcd)
        return (scaled_pcd / scaled_pcd_scale * scale) + pcd_center

    def __len__(self):
        return len(self.data_paths)

    @classmethod
    def get_validation_datasets(cls, root_dir, classes=[], **kwargs):
        raise NotImplementedError
    
    @classmethod
    def get_test_datasets(cls, root_dir, classes=[], **kwargs):
        return {'all': CustomDataset(root_dir=root_dir)}
