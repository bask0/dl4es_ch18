"""
Calculate metrics like correlation or rmse on multidimensional array along given dimentsions
using dask.

Metrics implemented:
* correlation           > xr_corr
* rmse                  > xr_rmse
* mean percentage error > xr_mpe
* bias                  > xr_bias
* modeling effficiency  > xr_mef

Only values present in both datasets are used to calculate metrics.

"""

import numpy as np
import xarray as xr
import warnings
from datetime import datetime


def percentile_gufunc(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        return np.nanpercentile(x, 0.5)


def xr_percentile(x, dim):
    p = 0.5
    m = xr.apply_ufunc(
        percentile_gufunc, x,
        input_core_dims=[[dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': f'{p}-percentile', 'units': x.attrs.get('units', '-')})
    m.name = f'{p}-percentile'
    return m


def pearson_cor_gufunc(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        valid_values = np.isfinite(x) & np.isfinite(y)
        valid_count = valid_values.sum(axis=-1)

        x[~valid_values] = np.nan
        y[~valid_values] = np.nan

        x -= np.nanmean(x, axis=-1, keepdims=True)
        y -= np.nanmean(y, axis=-1, keepdims=True)

        cov = np.nansum(x * y, axis=-1) / valid_count
        std_xy = (np.nanstd(x, axis=-1) * np.nanstd(y, axis=-1))

        corr = cov / std_xy

        return corr


def xr_corr(obs, mod, dim):
    m = xr.apply_ufunc(
        pearson_cor_gufunc, obs, mod,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'corr', 'units': '-'})
    m.name = 'corr'
    return m


def rmse_gufunc(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        se = np.power(x-y, 2)
        mse = np.nanmean(se, axis=-1)
        rmse = np.sqrt(mse)

        return rmse


def xr_rmse(obs, mod, dim):
    m = xr.apply_ufunc(
        rmse_gufunc, obs, mod,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'rmse'})
    m.name = 'rmse'
    return m


def mpe_gufunc(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mpe = 100 * np.nanmean((x-y)/x, axis=-1)

        return mpe


def xr_mpe(obs, mod, dim):
    m = xr.apply_ufunc(
        mpe_gufunc, obs, mod,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'mpe', 'units': '%'})
    m.name = 'mpe'
    return m


def bias_gufunc(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(x) & np.isfinite(y)

        x[~valid_values] = np.nan
        y[~valid_values] = np.nan

        return np.nanmean(x, axis=-1) - np.nanmean(y, axis=-1)


def xr_bias(obs, mod, dim):
    m = xr.apply_ufunc(
        bias_gufunc, obs, mod,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'bias'})
    m.name = 'bias'
    return m


def mef_gufunc(x, y):
    # x is obs, y is mod
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(x) & np.isfinite(y)

        x[~valid_values] = np.nan
        y[~valid_values] = np.nan

        sse = np.nansum(np.power(x-y, 2), axis=-1)
        sso = np.nansum(
            np.power(y-np.nanmean(y, axis=-1, keepdims=True), 2), axis=-1)

        mef = 1.0 - sse / sso

        return mef


def xr_mef(obs, mod, dim):
    m = xr.apply_ufunc(
        mef_gufunc, obs, mod,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'mef', 'units': '-'})
    m.name = 'mef'
    return m


def get_metric(mod, obs, fun, dim='time', verbose=False):
    """Calculate a metric along a dimension.

    Metrics implemented:
    * correlation           > xr_corr
    * rmse                  > xr_rmse
    * mean percentage error > xr_mpe
    * bias                  > xr_bias
    * modeling effficiency  > xr_mef

    Only values present in both datasets are used to calculate metrics.

    Parameters
    ----------
    data: xarray.Dataset
        Dataset with data variables 'mod' (modelled) and 'obs' (observed).
    fun: Callable
        A function that takes three arguments: Modelled (xarray.DataArray), observed (xarray.DataArray)
        and the dimension along which the metric is calculated.
    dim: str
        The dimension name along which the metri is calculated, default is `time`.

    Returns
    ----------
    xarray.Dataset

    """

    return fun(mod, obs, dim)


def get_metrics(mod, obs, funs, dim='time', verbose=True):
    """Calculate multiple metrics along a dimension and combine into single dataset.

    Metrics implemented:      name
    * correlation             > 'corr'
    * rmse                    > 'rmse'
    * mean percentage error   > 'mpe'
    * bias                    > 'bias'
    * modeling effficiency    > 'mef'

    Only values present in both datasets are used to calculate metrics.

    Parameters
    ----------
    mod: xarray.DataArray
        The modelled data.
    obs: xarray.DataArray
        The observed data.
    funs: Iterable[str]
        An iterable of function names (see `metrics implemented`).
    dim: str
        The dimension name along which the metri is calculated.
    verbose: bool
        Silent if False (True is default).

    Returns
    ----------
    xarray.Dataset

    """

    fun_lookup = {
        'corr': xr_corr,
        'rmse': xr_rmse,
        'mpe': xr_mpe,
        'bias': xr_bias,
        'mef': xr_mef
    }

    requested_str = ", ".join(funs)
    options_str = ", ".join(fun_lookup.keys())

    tic = datetime.now()

    if verbose:
        print(f'{timestr(datetime.now())}: calculating metrics [{requested_str}]')

    met_list = []
    for fun_str in funs:
        if verbose:
            print(f'{timestr(datetime.now())}: - {fun_str}')
        if fun_str not in fun_lookup:
            raise ValueError(
                f'Function `{fun_str}` not one of the implemented function: [{options_str}].'
            )
        fun = fun_lookup[fun_str]
        met_list.append(fun(mod, obs, dim).compute())

    met = xr.merge(met_list)

    toc = datetime.now()

    elapsed = toc - tic
    elapsed_mins = int(elapsed.seconds / 60)
    elapsed_secs = int(elapsed.seconds - 60 * elapsed_mins)

    if verbose:
        print(f'{timestr(datetime.now())}: done; elapsed time: {elapsed_mins} min {elapsed_secs} sec')

    return met


def timestr(t):
    return t.strftime("%m/%d/%Y, %H:%M:%S")
