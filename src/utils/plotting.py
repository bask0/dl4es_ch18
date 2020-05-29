
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

text_box = dict(facecolor='white', edgecolor='none', pad=6, alpha=.9)


def subplots_robinson(*args, **kwargs):
    return plt.subplots(*args, subplot_kw={'projection': ccrs.Robinson()}, **kwargs)


def subplots_plateCarree(*args, **kwargs):
    return plt.subplots(*args, subplot_kw={'projection': ccrs.PlateCarree()}, **kwargs)


def plot_map(
        ds,
        hist=True,
        label=None,
        coarsen=0,
        extent=[-180 + 1e-3, 180 - 1e-3, - 60 + 1e-3, 90 - 1e-3],
        histogram_placement=[0.04, 0.15, 0.23, 0.3],
        ax=None,
        figsize=(14, 7),
        subplot_kw={},
        hist_kw={},
        **kwargs):
    """Plot global map.

    Parameters
    ----------

    ds: xarray.Datarray
        Data to plot
    hist: bool
        If True (default), a histogram will be added to the map
    label: str
        The cbar label, default is 'None' (then dataset name or long_name is used).
    coarsen: int
        A factor by which the data is aggregated in both spatial dimensions. this can be usful if the resolution
        is very high and plotting takes too long.
    extent: iterable
        An iterable of length 4 that specifies plot extent, [lon min, lon max, lat min, lat max]. the default is
        [-180+1e-3, 180-1e-3, -60+1e-3, 90-1e-3] for global maps with Antarctica removed. If `None` is passed,
        xarrays default is used.
    histogram_placement: iterable
        Histogram placement relative to axes (0-1), an iterable of four: [x0, y0, width, height]
    ax: matplotlib axis, optional
        If passed, plot will be added to this axis. Else, a default axes will be used. Use `subplots_robinson(...)`
        to create subplots with Robinson projection equivalent to plt.subplots(...).
    figsize: (int, int), optional
        Figure size in inches, has no effect if `ax` is passed.
    subplot_kw: dict, optional
        Keyword arguments passed to subplot, has no effect if `ax` is passed.
    hist_kw: dict, optional
        Keyword arguments passed to plt.hist.
    kwargs:
        Keyword arguments passed to pcolormesh. Some useful kwargs:
        * cmap          str         -> Name of a colormap, e.g. 'viridis'
        * robust        bool        -> Cuts outliers in colormap
        * vmin, vmax    numeric     -> Set colormap min / max value
    """

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.subplot(projection=ccrs.Robinson(), **subplot_kw)

    if label is not None:
        ds.attrs['long_name'] = label
        ds.attrs['units'] = ''

    if coarsen > 0:
        coords = list(ds.coords)
        if 'lat' in coords:
            lat = 'lat'
        elif 'latitude' in coords:
            lat = 'latitude'
        else:
            raise ValueError(
                'Cannot infer latitude coorinates, must be `lat` or `latitude`.')
        if 'lon' in coords:
            lon = 'lon'
        elif 'longitude' in coords:
            lon = 'longitude'
        else:
            raise ValueError(
                'Cannot infer longitude coorinates, must be `lon` or `longitude`.')
        ds = ds.coarsen({lat: coarsen, lon: coarsen}).mean()

    ax.add_feature(cartopy.feature.LAND, facecolor='0.7')
    if ds.ndim == 2:
        cbar_kwargs = dict(
            orientation='horizontal',
            shrink=0.8,
            aspect=40,
            pad=0)
        if 'cbar_kwargs' in kwargs:
            cbar_kwargs.update(kwargs['cbar_kwargs'])
            kwargs.pop('cbar_kwargs')
        img = ds.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cbar_kwargs=cbar_kwargs,
            **kwargs)
    if ds.ndim == 3:
        hist = False
        img = ds.plot.imshow(
            ax=ax,
            transform=ccrs.PlateCarree(),
            **kwargs)
    ax.set_global()
    ax.coastlines()
    ax.gridlines()

    ax.xformatter = LONGITUDE_FORMATTER
    ax.yformatter = LATITUDE_FORMATTER

    if extent is not None:
        ax.set_extent(extent)

    if hist:
        axh = ax.inset_axes(histogram_placement)

        xmin, xmax = np.nanquantile(ds, [0.01, 0.99])

        if 'vmin' in kwargs:
            xmin = kwargs['vmin']
        if 'vmax' in kwargs:
            xmax = kwargs['vmax']

        if 'bins' not in hist_kw:
            hist_kw.update({'bins': 30})
        if 'color' not in hist_kw:
            hist_kw.update({'color': '0.5'})
        if 'range' not in hist_kw:
            hist_kw.update({'range': (xmin, xmax)})
        if 'edgecolor' not in hist_kw:
            hist_kw.update({'edgecolor': 'none'})

        ds.plot.hist(ax=axh, **hist_kw)

        axh.set_title('')
        axh.spines['right'].set_visible(False)
        axh.spines['left'].set_visible(False)
        axh.spines['top'].set_visible(False)
        axh.set_yticks([], [])
        axh.patch.set_facecolor('None')
        axh.set_xlabel('')

    return img


def plot_scatter(x, y, ax=None, subplot_kw={}, **kwargs):
    if ax is None:
        ax = plt.subplot(**subplot_kw)

    not_missing = x.notnull() & y.notnull()

    ax.scatter(x.where(not_missing, drop=True),
               y.where(not_missing, drop=True))


def plot_hist2d(x, y, ax=None, bins=60, xlabel='mod', ylabel='obs', title='', robust=0, subplot_kw={}, **kwargs):
    if ax is None:
        ax = plt.subplot(**subplot_kw)

    not_missing = x.notnull() & y.notnull()

    x_data = x.values[not_missing.data]
    y_data = y.values[not_missing.data]

    r = np.corrcoef(x_data, y_data)[0, 1]

    x_min, x_max = np.quantile(x_data, (robust, 1-robust))
    y_min, y_max = np.quantile(y_data, (robust, 1-robust))

    d_min = np.min((x_min, y_min))
    d_max = np.max((x_max, y_max))

    d_range = None
    if robust != 0:
        d_range = [[d_min, d_max], [d_min, d_max]]

    ax.hist2d(x_data, y_data, norm=mpl.colors.LogNorm(),
              bins=bins, range=d_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(d_min, d_max)
    ax.set_ylim(d_min, d_max)

    ax.plot((d_min, d_max), (d_min, d_max), '--', color='0.3')
    ax.text(0.05, 0.95, f'{title}  r={r:.2f}', horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes, bbox=text_box)


def plot_hexbin(
        x, y,
        xlabel='mod', ylabel='obs',
        robust=0,
        grindsize=120,
        title='',
        ax=None,
        subplot_kw={},
        **kwargs):

    if ax is None:
        ax = plt.subplot(**subplot_kw)

    not_missing = x.notnull() & y.notnull()

    x_data = x.values[not_missing.data]
    y_data = y.values[not_missing.data]

    r = np.corrcoef(x_data, y_data)[0, 1]
    mef = (1 - np.sum(np.power(x_data - y_data, 2)) / np.sum(np.power(y_data - np.mean(y_data), 2)))

    x_min, x_max = np.quantile(x_data, (robust, 1-robust))
    y_min, y_max = np.quantile(y_data, (robust, 1-robust))

    d_min = np.min((x_min, y_min))
    d_max = np.max((x_max, y_max))

    extent = [d_min, d_max, d_min, d_max]

    ax.hexbin(x, y, bins='log', gridsize=grindsize, mincnt=1, extent=extent, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(d_min, d_max)
    ax.set_ylim(d_min, d_max)

    ax.plot((d_min, d_max), (d_min, d_max), '--', color='0.3')
    ax.text(0.05, 0.95, f'{title}  r={r:.2f} mef={mef:.2f}', horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes, bbox=text_box)
