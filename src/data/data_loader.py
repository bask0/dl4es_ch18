import xarray as xr
import zarr
import numpy as np
import pandas as pd

from torch.utils.data.dataset import Dataset


class Data(Dataset):
    def __init__(
            self,
            config,
            partition_set):

        if partition_set not in ['train', 'valid', 'test']:
            raise ValueError(f'Argument `partition_set`: Mut be one of: train | valid | test.')

        self.dyn_features_names = config['dynamic_vars']
        self.dyn_features_path = config['dynamic_path']
        self.stat_features_names = config['static_vars']
        self.stat_features_path = config['static_path']
        self.dyn_target_name = config['target_var']
        self.dyn_target_path = config['target_path']
        self.mask_path = config['mask_path']

        self.dyn_features = zarr.open_group(self.dyn_features_path, mode='r')
        # self.stat_features = zarr.open_group(self.stat_features_path, mode='r')
        self.dyn_target = zarr.open_group(self.dyn_target_path, mode='r')

        # Date range, e.g. ['2000-01-01', '2004-12-31']
        self.range = config['time']['range']
        # Date range, e.g. [2000-01-01, 2000-01-02, ..., 2004-12-31]
        self.date_range = pd.date_range(
            self.range[0], self.range[1], freq='1D')

        # Date range, e.g. ['2000-01-01', '2004-12-31']
        self.part_range = config['time'][partition_set]

        # Apply warmup period:
        # - if training set: warmup period is added to training period start.
        # - if validation / testing: warmup period is subtracted from period start.
        if partition_set == 'train':
            warmup_start = self.part_range[0]
            warmup_end = f'{int(self.part_range[0][:4])+config["warmup"]}{self.part_range[0][4:]}'
        else:
            warmup_start = f'{int(self.part_range[0][:4])-config["warmup"]}{self.part_range[0][4:]}'
            warmup_end = self.part_range[0]

        # Indices of partition_set (e.g. training) reative to the dataset range.
        self.t_start = np.argwhere(self.date_range == warmup_start)[0][0]
        self.t_wamup_end = np.argwhere(self.date_range == warmup_end)[0][0]
        self.t_end = np.argwhere(self.date_range == self.part_range[1])[0][0]

        self.num_warmup_steps = self.t_wamup_end - self.t_start

        mask = xr.open_dataset(self.mask_path)
        self.coords = np.argwhere(mask.mask.values)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, inx):
        lat, lon = self.coords[inx]

        dyn_features = np.stack([
            (self.dyn_features[var][var][
                self.t_start:self.t_end, lat, lon]) for var in self.dyn_features_names
        ], axis=-1)

        dyn_target = self.dyn_target[self.dyn_target_name][self.t_start:self.t_end, lat, lon]

        return dyn_features, dyn_target
