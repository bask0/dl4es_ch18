
import json
import argparse
import os
import numpy as np
import xarray as xr
import time
import zarr

from utils.parallel import parcall


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_file',
        type=str,
        help='Configuration file (.json).',
        required=True
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        help='Output directory.'
    )

    parser.add_argument(
        '--bgi_path',
        type=str,
        help='BGI path.',
        default='/Net/Groups/BGI/'
    )

    parser.add_argument(
        '-o',
        '--overwrite',
        help='Overwtie existing variables, default is False.',
        action='store_true'
    )

    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.config_file):
        raise ValueError(
            'Configuration file (`file`) not found: ', args.config_file)
    if not os.path.isdir(args.bgi_path):
        raise ValueError('BGI path (`bgi_path`) not found: ', args.bgi_path)

    return args


def stack_data(files_in, file_out, overwrite):
    """Stack multiple years to one single netcdf file.

    Parameters
    ----------
    files_in        List of netcdf file paths.
    file_out        Output netcdf file path.
    overwrite       Bool, wheter to overwrite existing.

    """

    base_msg = f'Stacking to {file_out}... '
    tic = time.time()
    files = ' '.join(files_in)
    sys_call = f'cdo mergetime {files} {file_out}'

    print(base_msg)

    if os.path.exists(file_out):

        if overwrite:
            os.remove(file_out)
            os.system(sys_call)

        else:
            print(base_msg, 'exists, skipping.')

    else:
        os.system(sys_call)

    toc = time.time()
    elapsed = toc - tic
    print(base_msg, f'done, elapsed time: {elapsed/60:0.0f} min')


def write_zarr(zarr_file, file_in, varname, chunk_size):
    """Add xr.Dataset to zarr group.

    Parameters
    ----------
    zarr_file       Zarr group path.
    file_in         Input netcdf file path.
    varname         Name of ``xrdata`` variable to create in ``zarr_group``.
    chunk_size      Chunk size of lat / lon dimension.
    overwrite       Bool, wheter to overwrite existing.

    """

    base_msg = f'Converting to zarr: {varname}... '
    tic = time.time()
    print(base_msg)

    if os.path.exists(os.path.join(zarr_file, varname)):
        print(base_msg, 'exists, skipping.')

    else:
        ds = xr.open_dataset(file_in)

        if 'latitude' in ds.coords:
            ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        ds = ds.chunk({'time': -1, 'lat': chunk_size,
                       'lon': chunk_size})
        ds.to_zarr(
            zarr_file,
            group=varname,
            mode='w' if overwrite else 'a',
            encoding={varname: {'compressor': None}})

    toc = time.time()
    elapsed = toc - tic
    print(base_msg, f'done, elapsed time: {elapsed/60:0.0f} min')


def make_target_clean_again(file_in, file_out, year, overwrite):
    """Copy data and clean years.

    Parameters
    ----------
    files_in        List of netcdf file paths.
    file_out        Output netcdf file path.
    year            The year the dataset represents.
    overwrite       Bool, wheter to overwrite existing.

    """

    base_msg = f'Stacking to {file_out}... '
    tic = time.time()

    print(base_msg)

    ds = xr.open_dataset(file_in)

    if os.path.exists(file_out):

        if overwrite:
            os.remove(file_out)

            ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31')).to_netcdf(file_out)

        else:
            print(base_msg, 'exists, skipping.')

    else:
        ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31')).to_netcdf(file_out)

    toc = time.time()
    elapsed = toc - tic
    print(base_msg, f'done, elapsed time: {elapsed/60:0.0f} min')


if __name__ == '__main__':

    args = parse_args()

    config_file = args.config_file
    out_dir = args.out_dir
    bgi_path = args.bgi_path
    overwrite = args.overwrite

    os.makedirs(out_dir, exist_ok=True)

    with open(config_file) as f:
        config = json.load(f)

    years = np.arange(config['years'][0], config['years'][1] + 1)
    chunk_size = config['chunk_size']

    # Dynamic fearures --------------------------------------
    varnames = config['input']['dynamic']['varname']

    # Stack yearly datasets to one.
    data_path = config['input']['dynamic']['path']
    stack_dir = f'{out_dir}/org_data/gswp3_stacked/'
    os.makedirs(stack_dir, exist_ok=True)

    # Argument to parcall, needs to contain keywords of arguments to
    # `stack_data` with top-level items being iterated in parallel.
    par_args = {
        'files_in': [
            [
                os.path.join(bgi_path, f'{data_path}{var}/{var}.{y}.nc') for y in years
            ] for var in varnames
        ],
        'file_out': [
            os.path.join(stack_dir, f'{var}.nc') for var in varnames
        ]
    }

    parcall(iterable=par_args, fun=stack_data, num_cpus=2, overwrite=overwrite)

    # Convert stacks to zarr files.
    zarr_file = os.path.join(out_dir, 'input/dynamic/gswp3.zarr')
    os.makedirs(os.path.dirname(zarr_file), exist_ok=True)

    zarr.open_group(zarr_file, mode='w' if overwrite else 'a')

    par_args = {
        'file_in': [
            os.path.join(stack_dir, f'{var}.nc') for var in varnames
        ],
        'varname': varnames
    }

    parcall(par_args, fun=write_zarr, zarr_file=zarr_file, num_cpus=2, chunk_size=chunk_size)

    # Dynamic targets --------------------------------------
    varnames = config['target']['varname']

    # Stack yearly datasets to one.
    data_path = config['target']['path']
    stack_dir = f'{out_dir}/org_data/koirala2017/'

    os.makedirs(stack_dir, exist_ok=True)

    # Convert stacks to zarr files.
    zarr_file = os.path.join(out_dir, 'target/dynamic/koirala2017.zarr')
    os.makedirs(os.path.dirname(zarr_file), exist_ok=True)

    zarr.open_group(zarr_file, mode='w')

    # copying data to local disk.
    files_in = []
    files_out = []
    years_in = []

    for y in years:
        files_in.append(os.path.join(
            bgi_path, data_path.strip('/'), f'full_matsiro-gw_exp3_experiment_3_{y}.nc'))

        files_out.append(os.path.join(
            stack_dir, f'full_matsiro-gw_exp3_experiment_3_{y}.nc'))

        years_in.append(y)

    par_args = {
        'file_in': files_in,
        'file_out': files_out,
        'year': years_in
    }

    os.makedirs(os.path.dirname(files_out[0]), exist_ok=True)

    parcall(par_args, fun=make_target_clean_again,
            num_cpus=2, overwrite=overwrite)

    base_msg = f'Converting target... '

    tic = time.time()

    print(base_msg)

    ds = xr.open_mfdataset(files_out)[varnames]

    ds = ds.chunk({'time': -1, 'lat': chunk_size, 'lon': chunk_size})

    ds.to_zarr(
        zarr_file,
        mode='w',
        encoding={var: {'compressor': None} for var in varnames})

    toc = time.time()
    elapsed = toc - tic
    print(base_msg, f'done, elapsed time: {elapsed/60:0.0f} min')

    ds = xr.open_zarr(zarr_file)

    not_null = ds.notnull().all('time')
    varnames = list(not_null.data_vars)
    mask = not_null[varnames[0]]
    for v in varnames[1:]:
        mask = mask * not_null[v]

    mask = xr.Dataset({'mask': mask})

    mask.to_netcdf(os.path.join(out_dir, 'mask.nc'))
