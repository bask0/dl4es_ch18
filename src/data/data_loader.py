import os
import xarry as xr
import zarr
import numpy as np
import pandas as pd
import json

from torch.utils.data.dataset import Dataset


class Dataset(Dataset):
    def __init__(
            self,
            data_dir,
            config_file,
            partition_set):

        if not os.path.isdir(data_dir):
            raise ValueError(f'Argument `data_dir`: dir does not exist: {partition_set}')
        if not os.path.isfile(config_file):
            raise ValueError(f'Argument `config_file`: file does not exist: {config_file}')
        if partition_set not in ['train', 'valid', 'test']:
            raise ValueError(f'Argument `partition_set`: Mut be one of: train | valid | test.')

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

        # Date range, e.g. ['2000-01-01', '2004-12-31']
        self.range = config['time']['range']
        # Date range, e.g. [2000-01-01, 2000-01-02, ..., 2004-12-31]
        self.date_range = pd.date_range(self.time_range[0], self.time_range[1], freq='1D')

        # Date range, e.g. ['2000-01-01', '2004-12-31']
        self.part_range = config['time'][partition_set]

        # Indices of partition_set (e.g. training) reative to the dataset range.
        self.t_start = np.argwhere(self.date_range == self.part_range[0])[0][0]
        self.t_end = np.argwhere(self.date_range == self.part_range[1])[0][0]

        mask = xr.open_dataset(os.path.join(data_dir, 'mask.nc'))
        self.coords = np.argwhere(mask.mask.values)

    def __len__(self):
        self.coords.shape[0]

    def __getitem__(self, inx):
        lat, lon = self.coords[inx]

        dyn_features = np.stack([
            (self.dyn_features(var)[self.t_start:self.t_end, lat, lon]) for var in self.dyn_features_names
        ], axis=-1)

        dyn_target = self.dyn_target(self.dyn_target_name)[self.t_start:self.t_end, lat, lon]

        return dyn_features, dyn_target
