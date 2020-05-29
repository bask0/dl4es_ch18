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


def pearson_cor_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(mod) & np.isfinite(obs)
        valid_count = valid_values.sum(axis=-1)

        mod[~valid_values] = np.nan
        obs[~valid_values] = np.nan

        mod -= np.nanmean(mod, axis=-1, keepdims=True)
        obs -= np.nanmean(obs, axis=-1, keepdims=True)

        cov = np.nansum(mod * obs, axis=-1) / valid_count
        std_xy = (np.nanstd(mod, axis=-1) * np.nanstd(obs, axis=-1))

        corr = cov / std_xy

        return corr


def xr_corr(mod, obs, dim):
    m = xr.apply_ufunc(
        pearson_cor_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'corr', 'units': '-'})
    m.name = 'corr'
    return m


def rmse_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        se = np.power(mod-obs, 2)
        mse = np.nanmean(se, axis=-1)
        rmse = np.sqrt(mse)

        return rmse


def xr_rmse(mod, obs, dim):
    m = xr.apply_ufunc(
        rmse_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'rmse'})
    m.name = 'rmse'
    return m


def mpe_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mpe = 100 * np.nanmean((obs - mod) / obs, axis=-1)

        return mpe


def xr_mpe(mod, obs, dim):
    m = xr.apply_ufunc(
        mpe_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'mpe', 'units': '%'})
    m.name = 'mpe'
    return m


def bias_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(mod) & np.isfinite(obs)

        mod[~valid_values] = np.nan
        obs[~valid_values] = np.nan

        return np.nanmean(mod, axis=-1) - np.nanmean(obs, axis=-1)


def xr_bias(mod, obs, dim):
    m = xr.apply_ufunc(
        bias_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'bias'})
    m.name = 'bias'
    return m


def varerr_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(mod) & np.isfinite(obs)

        mod[~valid_values] = np.nan
        obs[~valid_values] = np.nan

        return np.square(mod.std(-1) - obs.std(-1))


def xr_varerr(mod, obs, dim):
    m = xr.apply_ufunc(
        varerr_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'varerr'})
    m.name = 'varerr'
    return m


def phaseerr_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(mod) & np.isfinite(obs)

        mod[~valid_values] = np.nan
        obs[~valid_values] = np.nan

        return (1.0 - pearson_cor_gufunc(mod, obs)) * 2.0 * mod.std(-1) * obs.std(-1)


def xr_phaseerr(mod, obs, dim):
    m = xr.apply_ufunc(
        phaseerr_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'phaseerr'})
    m.name = 'phaseerr'
    return m


def rel_bias_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(x) & np.isfinite(y)

        mod[~valid_values] = np.nan
        obs[~valid_values] = np.nan

        return (np.nanmean(mod, axis=-1) - np.nanmean(obs, axis=-1)) / np.nanmean(x, axis=-1)


def xr_rel_bias(obs, mod, dim):
    m = xr.apply_ufunc(
        bias_gufunc, obs, mod,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'relative bias'})
    m.name = 'rel_bias'
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


def get_metric(obs, mod, fun, dim='time', verbose=False):
    """Calculate a metric along a dimension.

    Metrics implemented:
    * correlation           > xr_corr
    * rmse                  > xr_rmse
    * mean percentage error > xr_mpe
    * bias                  > xr_bias
    * phaseerr              > xr_phaseerr
    * varerr                > xr_varerr
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

    return fun(obs, mod, dim)


def get_metrics(mod, obs, funs, dim='time', verbose=True):
    """Calculate multiple metrics along a dimension and combine into single dataset.

    Metrics implemented:      name
    * correlation           > xr_corr
    * rmse                  > xr_rmse
    * mean percentage error > xr_mpe
    * bias                  > xr_bias
    * phaseerr              > xr_phaseerr
    * varerr                > xr_varerr
    * modeling effficiency  > xr_mef

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
        'rel_bias': xr_rel_bias,
        'mef': xr_mef,
        'varerr': xr_varerr,
        'phaseerr': xr_phaseerr
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


def _single_xr_quantile(x, q, dim):
    if isinstance(dim, str):
        dim = [dim]
    ndims = len(dim)
    axes = tuple(np.arange(ndims)-ndims)
    m = xr.apply_ufunc(
        np.nanquantile, x,
        input_core_dims=[dim],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True,
        kwargs={'q': q, 'axis': axes})
    m.attrs.update({'long_name': f'{q}-quantile'})
    m.name = 'quantile'
    return m


def xr_quantile(x, q, dim):
    if not hasattr([1, 2], '__iter__'):
        q = [q]
    quantiles = []
    for i, q_ in enumerate(q):
        r = _single_xr_quantile(x, q_, dim).compute()
        quantiles.append(r)
    quantiles = xr.concat(quantiles, 'quantile')
    quantiles['quantile'] = q

    return quantiles


def global_cell_size(n_lat=180, n_lon=360, normalize=False, radius=6371.0, return_xarray=True):
    """Grid size per lon-lat cell on a sphere.
    
    Surface area (km^2) per grid cell for a longitude-latitude grid on a
    sphere with given radius. The grid is defined by the number of cells
    in latitude (n_lat) and longitude (l_lon). If normalize is True, the
    values get divided by the total area, such that the values can be
    directly used for weighting.
    
    Args:
        n_lat: int
            Size of latitude dimension.
        n_lon: int
            Size of latitude dimension.
        normalize: Bool (default: False)
            If True, the values get notmalized by the maximum area.
        radius: Numeric
            The radius of the sphere, default is 6371.0 for average Earth radius.
        return_xarray: Bool
            Wheter to return an xarray.DataArray or a numpy array. If True, the lat/lon
            coordinates are derived from n_lat / n_lon arguments, check code for details.
    Returns:
        2D numpy array or xrray.Dataset of floats and shape n_lat x n_lon, unit is km^2.
    
    """
    lon = np.linspace(-180., 180, n_lon+1)*np.pi/180
    lat = np.linspace(-90., 90, n_lat+1)*np.pi/180
    r = radius
    A = np.zeros((n_lat, n_lon))
    for lat_i in range(n_lat):
        for lon_i in range(n_lon):
            lat0 = lat[lat_i]
            lat1 = lat[lat_i+1]
            lon0 = lon[lon_i]
            lon1 = lon[lon_i+1]
            A[lat_i, lon_i] = (r**2.
                               * np.abs(np.sin(lat0)
                                         - np.sin(lat1))
                               * np.abs(lon0
                                         - lon1))
            
    gridweights = A / np.sum(A) if normalize else A
    if return_xarray:
        gridweights = xr.DataArray(gridweights,
                                   coords=[np.linspace(90, -90, n_lat*2+1)[1::2], np.linspace(-180, 180, n_lon*2+1)[1::2]],
                                   dims=['lat', 'lon'])
    return gridweights

def weighted_avg_and_std(xdata, weights=None):
    """
    Return the weighted average and standard deviation.

    Args:
        xdata : xr.DataArray
        weights : xr.DataArray, same shape as xdata

    Returns:
        (weighted_mean, weighted_std)
    """
    
    assert isinstance(xdata, xr.DataArray), 'xdata must be xr.DataArray'
    if weights is None:
        weights = xr.ones_like(xdata)
    assert isinstance(weights, xr.DataArray), 'weights must be xr.DataArray'
    assert xdata.shape == weights.shape, 'shape of xdata and weights must be equal'

    xdata = xdata.data
    weights = weights.data
    
    weights = weights[np.isfinite(xdata)]
    xdata = xdata[np.isfinite(xdata)]

    assert np.all(np.isfinite(weights)), 'Some weight are missing where xdata is not nan'

    if weights is None:
        weights = np.ones_like(xdata)
    if np.all(np.isnan(xdata)):
        average = np.nan
        variance = np.nan
    else:
        average = np.average(xdata, weights=weights)
        # Fast and numerically precise:
        variance = np.average((xdata-average)**2, weights=weights)
    return (average, np.sqrt(variance))