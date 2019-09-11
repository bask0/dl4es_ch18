
import argparse
import xarray as xr







def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        help='Configuration file (.json).',
        required=True
    )

    parser.add_argument(
        '--target_dir',
        type=str,
        help='Target directory relative to `bgi_path`.',
        default='work_3/dl_chapter14/'
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

    return args
