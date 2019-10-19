
from torch.utils.data.dataset import Dataset
import os
import xarry as xr
import zarr
import numpy as np
import json


class Dataset(Dataset):
    def __init__(
            self,
            data_dir,
            config_file,
            partition_set):

        self.data_dir = data_dir
        
        with open(config_file) as f:
            config = json.load(f)
        
        self.dyn_features_names = config['input']['dynamic']['varname']
        self.stat_features_names = config['input']['static']['varname']
        self.dyn_target_name = config['target']['varname']

        self.dyn_features_path = os.path.join(
            data_dir, 'input/dynamic/gswp3.zarr/')
        self.stat_features_path = os.path.join(
            data_dir, 'input/static/')
        self.dyn_target_path = os.path.join(
            data_dir, 'target/dynamic/koirala2017.zarr')

        self.dyn_features = zarr.open_group(self.dyn_features_path, mode='r')
        # self.stat_features = zarr.open_group(self.stat_features_path, mode='r')
        self.dyn_target = zarr.open_group(self.dyn_target_path, mode='r')

        self.years = config['years'][partition_set]
        self.years[1] = self.years[1] + 1

        mask = xr.open_dataset(os.path.join(data_dir, 'mask.nc'))
        self.coords = np.argwhere(mask.mask.values)

    def __len__(self):
        self.coords.shape[0]

    def __getitem__(self, inx):
        lat, lon = self.coords[inx]

        data = np.stack([
            (self.dyn_features(var)[t0:t1, ..., lat, lon]) for var in self.dyn_features_names
        ], axis=-1)
