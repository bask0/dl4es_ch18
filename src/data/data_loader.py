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
    used for training and, shifted by 6 pixels, for validation. If 'is_tune' is 'False',
    all locations where mask == fold are sampled for tesing while all other valid
    locations (mask != 0 & mask != fold) are used for training.

    The temporal partition (defined in the config file) is fixed, 'partition_sets' defines
    whether training or validation period is selected.

    Parameters
    ----------
    config (dict):
        Configuration.
    partition_set (str):
        Cross validation partition set, one of 'train' | 'eval'.
    fold (int):
        The spatial cross-validation fold. Must correspond to values in the mask. -1 means that
        ll data is used. See 'Notes'.
        for more details.
    is_tune (bool):
        If 'True', a subset fo the data will be sampled for tuning of hyperparameters. See
        'Notes' for more details. If 'True', the 'fold' argument has no effect.
    small_aoi: bool
        If True, a subset of the data is used (Europe and Western Asia).

    Returns
    ----------
    features_d: nd array
        Dynamic features with shape <time, num_features>
    features_s: nd array
        Static features with shape <1, num_features>
    target: nd array
        Dynamic target with shape <time, 1>
    (lat, lon): tuple of integers
        Latitude and longitude coordinates

    """

    def __init__(
            self,
            config,
            partition_set,
            fold=None,
            is_tune=False,
            small_aoi=False):

        if partition_set not in ['train', 'eval']:
            raise ValueError(
                f'Argument `partition_set`: Must be one of: [`train` | `eval`].')

        def msg(
            x): return f'Argument ``{x}`` is not an iterable of string elements.'
        if not self._check_striterable(config['input_vars']):
            raise ValueError(msg('input_vars'))
        if not self._check_striterable(config['input_vars_static']):
            raise ValueError(msg('input_vars_static'))
        if not isinstance(config['target_var'], str):
            raise ValueError('Argument ``target_var`` must be a string.')

        self.input_vars = config['input_vars']
        self.input_vars_static = config['input_vars_static']
        self.target_var = config['target_var']

        self.partition_set = partition_set
        self.fold = fold
        self.is_tune = is_tune
        self.small_aoi = small_aoi
        # self.permute_time = permute_time if partition_set == 'train' else False

        self.config = config
        self.data_path = self.config['data_path']

        ds = xr.open_zarr(self.data_path)
        self.dynamic_vars, self.static_vars = self._get_static_and_dynamic_varnames(ds)

        self.time_slicer = TimeSlice(
            ds_path=self.data_path,
            date_range=config['time'][partition_set],
            warmup=config['time']["warmup"],
            partition_set=partition_set,
            train_seq_length=config['time']["train_seq_length"] if partition_set == 'train' else 0)

        self.num_warmup_steps = self.time_slicer.num_warmup

        mask = ds['mask']

        if small_aoi:
            # Reduce AOI size.
            print('Test run: training on lat > 0 & lon > 0')
            mask = mask.where((
                mask.lat > 35) & (
                mask.lat < 70) & (
                mask.lon > 0) & (
                mask.lon < 120
            ), 0, drop=False)

        # The mask contains 0 for non-valid pixels and integers > 0 for the folds. Here, we get
        # all fold integers.
        folds = np.setdiff1d(np.unique(mask), 0)
        if any(folds < 0):
            raise ValueError(
                f'The mask cannot contain values < 0.')

        # For HP tuning, a subset of the data is used, every 3rd pixel in lat / lon direction.
        if is_tune:

            sparse_grid = self._get_sparse_grid(mask, 3)

            mask = (mask == 1).astype(int)

            mask *= sparse_grid

        # For other model runs, the fold is used for validationg, while all other folds are
        # used for training. E.g. if folds are [1, 2, 3] and fold is 1, the folds [2, 3] are
        # for training and 1 for validation.
        else:

            if (fold < -1) or (fold == 0):
                raise ValueError(
                    f'Argument `fold` must be -1 or a value > 0 but is {fold}.')

            if fold not in np.append(folds, -1):
                raise ValueError(
                    f'Fold `{fold}` not found in mask with unique values `{folds}`.')

            # Select all folds.
            if fold == -1:
                mask_select = folds
            else:
                if len(folds) == 1:
                    raise ValueError(
                        f'As the mask contains only one fold ({folds}), you must pass '
                        '``fold=-1`` to make training and valiation both use the same '
                        'fold.'
                    )
                if partition_set == 'train':
                    mask_select = np.setdiff1d(folds, fold)
                else:
                    mask_select = [fold]

            mask = mask.isin(mask_select)

        self.mask = mask
        self.coords = np.argwhere(mask.values)

        self.ds_stats = {
            var: {
                'mean': np.float32(ds[var].attrs['mean']),
                'std': np.float32(ds[var].attrs['std'])
            } for var in np.setdiff1d(ds.data_vars, 'mask')
        }

        self.ds = zarr.open(self.data_path, mode='r')
        self._check_all_vars_present_in_dataset()
        self._check_var_time_dim()

        # Get classification dataset stats for one-hot encoding.
        num_static = 0
        self._static_class_stats = {}
        for var in self.input_vars_static:
            s = self.ds[var]
            min_class = int(np.nanmin(s))
            max_class = int(np.nanmax(s))
            num_classes = max_class - min_class + 1
            num_static += num_classes
            self._static_class_stats.update({
                var: {
                    'min': min_class,
                    'num_classes': num_classes
                }
            })

        self.num_dynamic = len(self.input_vars)
        self.num_static = num_static

    def __len__(self):
        return self.coords.shape[0]

    def one_hot(self, val, var):
        enc = np.zeros(
            self._static_class_stats[var]['num_classes'], dtype=np.float32)
        enc[int(val) - self._static_class_stats[var]['min']] = 1
        return enc

    def _sel_time(self, var, t_start, t_end, lat, lon):
        """Select time range and pixel from data.

        If the variable is soil moisture (SM) related, the time index
        is shifted one to the past, as SM reflects the state at the
        end of the day while fluxes like ET are integrated over a
        day. SM at the end of the day cannot impact the fluxes during
        the day and thus, the entire SM time-series is shifted such
        that instead of  sm_2 -> et_2, we get sm_1 -> et_2.
        """
        if var in ('mrlsl_shal', 'mrlsl_deep'):
            t_start -= 1
            t_end -= 1

        return self.ds[var][t_start:t_end, lat, lon]

    def __getitem__(self, inx):
        lat, lon = self.coords[inx]

        t_start, t_end = self.time_slicer.get_time_range()

        # Each single temporal variable has shape <time, lat, lon>. We select one coordinate, yielding
        # shape <time>. All variables are then stacked along last dimension, yielding <time, num_vars>
        features_d = np.stack(
            [self.standardize(self._sel_time(var, t_start, t_end, lat, lon), var)
             for var in self.input_vars],
            axis=-1
        )

        # Each single non-temporal variable has shape <lat, lon>. We select one coordinate, yielding
        # shape <> (scalar) and apply one-hot encoding. All variables are then concatenated into one
        # vector and expanded in the first diension, yielding <1, num_classes>.
        features_s = np.concatenate(
            [self.one_hot(self.ds[var][lat, lon], var)
             for var in self.input_vars_static],
            axis=0
        ).reshape(1, -1)
        #features_s = features_s.repeat(features_d.shape[0], axis=0)
        #features_d = np.concatenate((features_d, features_s), axis=-1)

        # The (temporal) target variable has shape <time, lat, lon>. We select one coordinate, yielding
        # shape <time>, and expand in the last dimension, yielding <time, 1>.
        target = self.standardize(
            self.ds[self.target_var][t_start:t_end, lat, lon], self.target_var).reshape(-1, 1)

        if np.any(np.isnan(features_d)):
            raise ValueError('NaN in features, training stopped.')

        # Random permute time-series.
        #if self.permute_time:
        #    perm_indx = torch.randperm(features_d.shape[0])
        #    features_d = features_d[perm_indx, :]
        #    target = target[perm_indx, :]

        return features_d, features_s, target, (lat, lon)

    def get_empty_xr(self):
        ds = xr.open_zarr(self.data_path)[self.target_var].sel(
                time=slice(self.time_slicer.seldate_first, self.time_slicer.seldate_last))

        return ds

    def standardize(self, x, varname):
        return (x - self.ds_stats[varname]['mean']) / self.ds_stats[varname]['std']

    def unstandardize(self, x, varname):
        return x * self.ds_stats[varname]['std'] + self.ds_stats[varname]['mean']

    def _check_striterable(self, x):
        is_iterable_non_str = hasattr(x, '__iter__') & (not isinstance(x, str))
        all_elements_are_str = all([isinstance(x_, str) for x_ in x])
        return is_iterable_non_str & all_elements_are_str

    def _check_all_vars_present_in_dataset(self):
        def msg(
            x): return f'Variable ``{x}`` not found in dataset located at {self.config["path"]}.'

        for var in self.input_vars + self.input_vars_static + [self.target_var]:
            if var not in self.ds:
                raise ValueError(msg(var))

    def _check_var_time_dim(self):
        def msg(x, y):
            return f'Variable ``{x}`` seems to be {"non-" if y else ""}temporal, check the variable arguments.'

        for var in self.input_vars + [self.target_var]:
            if self.ds[var].ndim != 3:
                raise ValueError(msg(var, True))
        for var in self.input_vars_static:
            if self.ds[var].ndim != 2:
                raise ValueError(msg(var, False))

    def _get_static_and_dynamic_varnames(self, ds):
        time_vars = []
        non_time_vars = []
        for var in ds.data_vars:
            if 'time' in ds[var].dims:
                time_vars.append(var)
            else:
                non_time_vars.append(var)
        return time_vars, non_time_vars

    def _get_sparse_grid(self, x, gap_size):
        nlat = len(x.lat)
        nlon = len(x.lon)
        r = np.zeros((nlat, nlon), dtype=int)

        for lat in np.arange(0, nlat - 1 - gap_size, gap_size * 2):
            for lon in np.arange(0, nlon - 1 - gap_size, gap_size * 2):
                r[lat, lon] = 1
                r[lat + gap_size, lon + gap_size] = 2

        m = xr.DataArray(r, coords=(x.lat, x.lon))

        return m


class TimeSlice(object):
    """Manage time slicing for training and evaluation set.

    Parameters
    ----------
    ds_path: str
        Path to the data cube (.zarr format).
    data_range: tuple(str, str)
        Date range to read, e.g. ('2000-01-01', '2005-12-31')
    warmup: int
        Number of warmup years that are not used in loss functino. For the training set,
        the warmup period is added **after** the lower time bound, for the evalation set it
        is added **before** the lower time bound, overlapping with the training time-range.
    partition_set: str
        One of ['train' | 'eval']
    train_seq_len: int
        If this is not 0 (default), a sequence of this length will be sampled from the time
        range. the warmup period remains the same, e.g. for a warmup period of one year and
        ``train_seq_len=100``, a time range of warmup + 100 is randomly selected when calling
        ``TimeSlice:get_time_range``.

    """

    def __init__(
            self,
            ds_path,
            date_range,
            warmup,
            partition_set,
            train_seq_length):

        if partition_set not in ['train', 'eval']:
            raise ValueError(
                f'Argument `partition_set`: Must be one of: [`train` | `eval`].')

        self.partition_set = partition_set
        self.do_seq_sample = train_seq_length != 0

        ds_time = xr.open_zarr(ds_path).time
        date_first = pd.to_datetime(ds_time.values[0])
        date_last = pd.to_datetime(ds_time.values[-1])

        seldate_first = pd.to_datetime(date_range[0])
        seldate_last = pd.to_datetime(date_range[1])

        if not (date_first <= seldate_first < date_last):
            raise ValueError(
                f'The selected lower time-series bound ({seldate_first}) '
                f'is not in the dataset time range ({date_first} - {date_last})'
            )
        if not (date_first < seldate_last <= date_last):
            raise ValueError(
                f'The selected upper time-series bound ({seldate_last}) '
                f'is not in the dataset time range ({date_first} - {date_last})'
            )

        warmup_delta = pd.DateOffset(years=warmup)

        # warmup_first is the actual lower limit after applying the warmup period
        # seldate_first is the first date after the warmup period
        if partition_set == 'eval':
            warmup_first = seldate_first - warmup_delta
            if not (date_first <= warmup_first < date_last):
                raise ValueError(
                    f'After applying the warmup period of {warmup} year(s), the lower time-series '
                    f'bound ({warmup_first}) is not in the dataset time range ({date_first} - {date_last}). '
                    f'Note that the warmup period is added before the lower date range in the eval set.'
                )
        else:
            warmup_first = seldate_first
            seldate_first += warmup_delta
            if not (date_first < seldate_first <= date_last):
                raise ValueError(
                    f'After applying the warmup period of {warmup} year(s), the upper time-series '
                    f'bound ({seldate_first}) is not in the dataset time range ({date_first} - {date_last}). '
                    f'Note that the warmup period is added after the lower date range in the train set.'
                )

        self.warmup_first = warmup_first
        self.seldate_first = seldate_first
        self.seldate_last = seldate_last

        date_range = pd.date_range(date_first, date_last)

        self.num_warmup = (seldate_first - warmup_first).days
        self.train_seq_length = self.num_warmup + train_seq_length
        self.start_t = np.argwhere(warmup_first == date_range).item()
        self.end_t = np.argwhere(seldate_last == date_range).item() + 1
        self.seq_len = self.end_t - self.start_t

        if self.seq_len < self.train_seq_length:
            raise ValueError(
                f'The sequence length (train_seq_length ({train_seq_length}) + warmup ({self.num_warmup}) = '
                f' {self.train_seq_length}) is longer than the sequence length ({self.seq_len}).')

        # This is used if train_seq_length > 0 to randomly select a time range of this length from the sequence.
        self.sample_range = range(
            self.start_t, self.end_t - self.train_seq_length)

    def get_time_range(self):
        if not self.do_seq_sample:
            return self.start_t, self.end_t
        else:
            start_t = np.random.choice(self.sample_range)
            end_t = start_t + self.train_seq_length
            return start_t, end_t

    def __repr__(self):
        s = (
            f'TimeSlice object\n\n'
            f'Partition set: {self.partition_set}\n'
            f'Sample lenth: {self.train_seq_length-self.num_warmup if self.do_seq_sample else "full sequence"}d\n\n'
            f'   warmup period: {self.num_warmup:5d}d        sample period: {self.seq_len-self.num_warmup:5d}d\n'
            f'|-------------------------|------------------------------|\n'
            f'|                         |                              | \n'
            f'{self.warmup_first.strftime("%Y-%m-%d")}            '
            f'{self.seldate_first.strftime("%Y-%m-%d")}                '
            f'{self.seldate_last.strftime("%Y-%m-%d")}\n\n'
            f'warmup start: {self.warmup_first.strftime("%Y-%m-%d")}\n'
            f'sample start: {self.seldate_first.strftime("%Y-%m-%d")}\n'
            f'sample end:   {self.seldate_last.strftime("%Y-%m-%d")}\n'

        )
        return s
