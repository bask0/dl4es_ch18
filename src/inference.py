import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--summary_dir',
        '-s',
        type=str,
        help='Path to summary dir containing model configuration and checkpoint.',
        default='default'
    )

    parser.add_argument(
        '--overwrite',
        '-O',
        help='Flag to overwrite existing runs (all existng runs will be lost!).',
        action='store_true'
    )

    args = parser.parse_args()

    return args



