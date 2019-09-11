
import json
import argparse
import os
import numpy as np
import xarray as xr
import time
import zarr
import shutil
import ray

from utils.parallel import parcall


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--congig_file',
        type=str,
        help='Configuration file (.json).',
        required=True
    )

    parser.add_argument(
        '--target_dir',
        type=str,
        help='Target directory relative to `bgi_path`.'
    )

    parser.add_argument(
        '--bgi_path',
        type=str,
        help='BGI path.',
        default='/Net/Groups/BGI/'
    )

    parser.add_argument(
        '-o'
        '--overwrite',
        type=bool,
        help='Overwtie existing variables, default is False.',
        action='store_false'
    )

    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.file):
        raise ValueError('Configuration file (`file`) not found: ', args.file)
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

    base_msg = 'Stacking to {target}... '
    tic = time.time()
    files = ' '.join(files_in)
    sys_call = f'cdo mergetime {files} {file_out}'

    print(base_msg)

    if os.path.isfile(file_out):

        if overwrite:
            os.remove(file_out)
            os.system(sys_call)

        else:
            print(base_msg, 'exists, skipping.')

    else:
        os.system(sys_call)

    toc = time.time()
    elapsed = tic - toc
    print(base_msg, f'done, elapsed time: {elapsed/60:0.0f} min')

def write_zarr(zarr_file, file_in, varname, chunk_size, overwrite):
    """Add xr.Dataset to zarr group.

    Parameters
    ----------
    zarr_file       Zarr group path.
    file_in         Input netcdf file path.
    varname         Name of ``xrdata`` variable to create in ``zarr_group``.
    chunk_size      Chunk size of lat / lon dimension.
    overwrite       Bool, wheter to overwrite existing.

    """

    base_msg = 'Converting to zarr: {varname}... '
    tic = time.time()
    print(base_msg)

    ds = xr.open_dataset(file_in)
    ds = ds.chunk({'time': -1, 'latitude': chunk_size, 'longitude': chunk_size})
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds.to_zarr(
        zarr_file,
        group=varname,
        mode='w' if overwrite else 'a',
        encoding={varname: {'compressor': None}})

    toc = time.time()
    elapsed = tic - toc
    print(base_msg, f'done, elapsed time: {elapsed/60:0.0f} min')



if __name__ == '__main__':

    args = parse_args()

    congig_file = args.congig_file
    target_dir = args.target_dir
    bgi_path = args.bgi_path
    overwrite = args.overwrite

    os.makedirs(target_dir, exist_ok=True)

    with open(congig_file) as f:
        config = json.load(f)

    years = np.arange(config['years'][0], config['years'][1] + 1)
    chunk_size = config['chunk_size']

    # Dynamic fearures --------------------------------------
    varnames = config['input']['dynamic']['varname']

    # Stack yearly datasets to one.
    data_path = config['input']['dynamic']['path']
    stack_dir = f'{target_dir}/org_data/gswp3_stacked/'
    os.makedirs(stack_dir, exist_ok=True)

    # Argument to parcall, needs to contain keywords of arguments to
    # `stack_data` with top-level items being iterated in parallel.
    par_args = {
        'files_in': [
            [
                bgi_path + f'{data_path}{var}/{var}.{y}.nc' for y in years
            ] for var in varnames
        ],
        'file_out': [
            os.path.join(stack_dir, f'{var}.nc') for var in varnames
        ]
    }

    parcall(iterable=par_args, fun=stack_data, num_cpus=2, overwrite=overwrite)

    # Convert stacks to zarr files.
    zarr_file = os.path.join(target_dir, '/input/dynamic/gswp3.zarr')
    os.makedirs(os.path.dirname(zarr_file), exist_ok=True)
    zarr.open_group(zarr_file, mode='w' if overwrite else 'a')

    par_args = {
        'file_in': [
            os.path.join(stack_dir, f'{var}.nc') for var in varnames
        ],
        'varname': varnames
    }

    parcall(par_args, fun=write_zarr, zarr_file=zarr_file, num_cpus=2, overwrite=overwrite)
