import xarray as xr
import zarr
import numpy as np
import pandas as pd

from torch.utils.data.dataset import Dataset


class Data(Dataset):
    """Data loader for simulated hydrological data (koirala2017).

    Notes
    ----------
    A mask is used to define valid pixels to sample from. 0 corresponds to non-valid
    pixels (e.g. ocean) and values > 0 are used to defiene spatial sampling locations
    of the cross-validation folds. If 'is_tune' is set to 'True', all pixels where
    mask > 0 are sample locations, but only every 12th pixel in both directions is
    used for training and, shifted by 6 pixels, for training. If 'is_tune' is 'False',
    all locations where mask == fold are sampled for tesing while all other valid
    locations (mask != 0 & mask != fold) are used for training.

    The temporal partition (defined in the config file) is fixed, 'partition_sets' dfines
    whether traiing or validation period is selected.


    Parameters
    ----------
    config (str):
        Path to configuration file.
    partition_set (str):
        Cross validation partition set, one of 'train' | 'eval'.
    fold (int):
        The spatial cross-validation fold. Must correspond to values in the mask. See 'Notes'
        for more details.
    is_tune (bool):
        If 'True', a subset fo the data will be sampled for tuning of hyperparameters. See
        'Notes' for more details. In this case, the 'fold' argument has no effect.

    """

    def __init__(
            self,
            config,
            partition_set,
            fold=None,
            is_tune=False):

        if partition_set not in ['train', 'eval']:
            raise ValueError(
                f'Argument `partition_set`: Mut be one of: train | eval.')

        if is_tune ^ (fold is None):
            raise ValueError(
                'Either pass argument `fold` OR set `is_tune=True`.')

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
        # - if evaludation set: warmup period is subtracted from period start.
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

        mask = mask.mask
        folds = np.setdiff1d(np.unique(mask), 0)

        if any(folds < 0):
            raise ValueError(
                f'The mask ({self.mask_path}) cannot contain values < 0.')

        if is_tune:

            sparse_grid = get_sparse_grid(mask, 6)

            mask = (mask > 0).astype(int)

            mask *= sparse_grid

            if partition_set == 'train':
                mask_select = [1]
            else:
                mask_select = [2]

        else:

            if fold <= 0:
                raise ValueError(
                    f'Argument `fold` must be a values > 0 but is {fold}.')

            if fold not in folds:
                raise ValueError(
                    f'Fold `{fold}` not found in mask with unique values `{folds}`.')

            if partition_set == 'train':
                mask_select = np.setdiff1d(np.unique(mask), fold)
            else:
                mask_select = [fold]

        self.coords = np.argwhere(np.isin(mask, mask_select))

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, inx):
        lat, lon = self.coords[inx]

        dyn_features = np.stack([
            (self.dyn_features[var][var][
                self.t_start:self.t_end, lat, lon]) for var in self.dyn_features_names
        ], axis=-1)

        dyn_target = self.dyn_target[self.dyn_target_name][self.t_start:self.t_end, lat, lon]

        return dyn_features, dyn_target, (lat, lon)

    def get_empty_xr(self):
        ds = xr.open_zarr(
            self.dyn_target_path)[[self.dyn_target_name]].isel(
                time=slice(self.t_start + self.num_warmup_steps, self.t_end))
        ds[self.dyn_target_name].values = np.nan
        ds[self.dyn_target_name + '_obs'] = ds[self.dyn_target_name]
        return ds


def get_sparse_grid(x, gap_size):
    nlat = len(x.lat)
    nlon = len(x.lon)
    r = np.zeros((nlat, nlon), dtype=int)

    for lat in np.arange(0, nlat, gap_size * 2):
        for lon in np.arange(0, nlon, gap_size * 2):
            r[lat, lon] = 1
            r[lat + gap_size, lon + gap_size] = 2

    m = xr.DataArray(r, coords=(x.lat, x.lon))

    return m
