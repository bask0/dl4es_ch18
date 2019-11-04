"""
Analyze a ray.tune run; plot training curves, show best run etc.
"""

import argparse
from typing import Dict, Any
from ray.tune.analysis import ExperimentAnalysis
import os
import shutil
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TRAIN_METRIC = 'loss_train'
DEFAULT_VALID_METRIC = 'loss_eval'
BASE_PATH = '/scratch/dl_chapter14/experiments/'
DEFAULT_TARGET_BASE_DIR = '/workspace/bkraft/dl_chapter14/experiments/'


def parse_args() -> Dict[str, Any]:
    """Parse arguments.

    Returns
    --------
    Dict of arguments.

    """
    parser = argparse.ArgumentParser(
        description=(
            'Summarize ray.tune runs from a given experiment state file (.json). A directory called '
            '`summary` is created in the directory containing the .json file.'
        ))

    parser.add_argument(
        '-p',
        '--path',
        type=str,
        help='experiment state file (.json)'
    )

    parser.add_argument(
        '--cp_dir',
        type=str,
        default='None',
        help='directory to copy summary to; If `None`, summary will not be copied'
    )

    parser.add_argument(
        '-I',
        '--infer_cp_dir',
        action='store_true',
        help='if `true`, the cp_dir will be infered'
    )

    parser.add_argument(
        '-t',
        '--train_metric',
        type=str,
        default=DEFAULT_TRAIN_METRIC,
        help='the train metric name'
    )

    parser.add_argument(
        '-e',
        '--eval_metric',
        type=str,
        default=DEFAULT_VALID_METRIC,
        help='the evaluation metric name (also metric that is used to find best run)'
    )

    parser.add_argument(
        '-O',
        '--overwrite',
        action='store_true',
        help='whether to overwrite existing out_dir, default is `false`'
    )

    args = parser.parse_args()

    return args


def summarize(
        path: str,
        cp_dir: str = 'None',
        infer_cp_dir: bool = False,
        train_metric: str = DEFAULT_TRAIN_METRIC,
        eval_metric: str = DEFAULT_VALID_METRIC,
        overwrite: bool = False,
        return_analysis: bool = False) -> None:

    print(f'\nLoading experiment state file loaded from:  \n{path}\n')

    if not os.path.isfile(path):
        raise ValueError(f'Path does not exist or is directory:\n{path}')
    if path[-5:] != '.json':
        raise ValueError(f'Not a .json file:\n{path}')

    summary_dir = os.path.join(os.path.dirname(
        os.path.dirname(path)), 'summary')

    if os.path.isdir(summary_dir):
        if not overwrite:
            raise ValueError(
                f'Target directory `{summary_dir}` exists, use `--overwrite` to replace.')
        shutil.rmtree(summary_dir)
    os.makedirs(summary_dir)

    path_split = path.split(BASE_PATH)[1].split('/')
    experiment = path_split[0]
    name = path_split[1]
    mode = path_split[2]

    if infer_cp_dir:
        cp_dir = os.path.join(
            DEFAULT_TARGET_BASE_DIR,
            experiment,
            name,
            mode,
            'summary'
        )
        print(f'\nInfering cp_dir:\n  {cp_dir}\n')
    if cp_dir != 'None':
        if cp_dir[-1] == '/':
            cp_dir = cp_dir[:-1]
        cp_dir_base = os.path.dirname(cp_dir)
        if os.path.isdir(cp_dir):
            shutil.rmtree(cp_dir)
        os.makedirs(cp_dir_base, exist_ok=True)

    exp = ExperimentAnalysis(path)

    if return_analysis:
        return exp

    configs = exp.dataframe()
    configs['rundir'] = [os.path.join(l, 'progress.csv')
                         for l in configs['logdir']]
    runs = []
    for i, f in enumerate(configs['rundir']):
        df = pd.read_csv(f)
        df['uid'] = i
        runs.append(df)
    runs = pd.concat(runs)

    best_run_dir = exp.get_best_logdir(eval_metric, mode='min')
    best_run_file = os.path.join(best_run_dir, 'progress.csv')
    best_run = df = pd.read_csv(best_run_file)

    print(f'Best run ID: {best_run_dir}')

    for f in ['json', 'pkl']:
        in_file = os.path.join(best_run_dir, f'params.{f}')
        out_file = os.path.join(summary_dir, f'best_params.{f}')
        copyfile(in_file, out_file)

    # Plot runs.
    plot_all(runs, eval_metric, os.path.join(summary_dir, 'all_runs.png'))
    plot_single(best_run, eval_metric, os.path.join(
        summary_dir, 'best_run.png'))

    if cp_dir != 'None':
        shutil.copytree(summary_dir, cp_dir)


def plot_all(runs: pd.core.frame.DataFrame, metric: str, savepath: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(
        10, 8), sharex=True, sharey='row', gridspec_kw={'wspace': 0, 'hspace': 0})
    box = dict(facecolor='yellow', pad=6, alpha=0.2)

    ax[0].text(
        1.0, 1.0, 'HYPERBAND OPTIMIZATION', transform=ax[0].transAxes,
        horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
    ax[0].text(
        0.5, 0.95, 'TRAINING', transform=ax[0].transAxes,
        horizontalalignment='center', verticalalignment='top', bbox=box)
    ax[1].text(
        0.5, 0.95, 'VALIDATION', transform=ax[1].transAxes,
        horizontalalignment='center', verticalalignment='top', bbox=box)

    train_name = DEFAULT_TRAIN_METRIC
    valid_name = DEFAULT_VALID_METRIC

    runs.groupby(['uid']).plot(
        x='epoch', y=train_name, ax=ax[0], legend=False)
    runs.groupby(['uid']).plot(
        x='epoch', y=valid_name, ax=ax[1], legend=False)

    ymin = np.min((
        np.min(runs[train_name]),
        np.min(runs[valid_name]))) * 0.9
    ymax = np.max(
        (np.percentile(runs[train_name], 95), np.percentile(runs[valid_name], 95)))
    xmin = np.min(runs['epoch'])-np.max(runs['epoch'])*0.01
    xmax = np.max(runs['epoch'])*1.01

    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(ymin, ymax)
    ax[0].yaxis.set_label_coords(-0.15, 0.5, transform=ax[0].transAxes)
    ax[0].set_ylabel('loss', bbox=box)

    fig.savefig(savepath, bbox_inches='tight', dpi=200, transparent=True)


def plot_single(single_run: pd.core.frame.DataFrame, metric: str, savepath: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(
        10, 8), sharex=True, sharey='row', gridspec_kw={'wspace': 0, 'hspace': 0})
    box = dict(facecolor='yellow', pad=6, alpha=0.2)

    ax[0].text(
        1.0, 1.0, 'BEST RUN', transform=ax[0].transAxes,
        horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
    ax[0].text(
        0.5, 0.95, 'TRAINING', transform=ax[0].transAxes,
        horizontalalignment='center', verticalalignment='top', bbox=box)
    ax[1].text(
        0.5, 0.95, 'EVALUATION', transform=ax[1].transAxes,
        horizontalalignment='center', verticalalignment='top', bbox=box)

    train_name = DEFAULT_TRAIN_METRIC
    valid_name = DEFAULT_VALID_METRIC

    single_run.plot(x='epoch', y=train_name, ax=ax[0], legend=False)
    single_run.plot(x='epoch', y=valid_name, ax=ax[1], legend=False)

    ymin = np.min((np.min(single_run[train_name]), np.min(
        single_run[valid_name]))) * 0.95
    ymax = np.max((np.percentile(single_run[train_name], 95), np.percentile(
        single_run[valid_name], 95)))
    xmin = np.min(single_run['epoch'])-np.max(single_run['epoch'])*0.01
    xmax = np.max(single_run['epoch'])*1.01

    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(ymin, ymax)
    ax[0].yaxis.set_label_coords(-0.15, 0.5, transform=ax[0].transAxes)
    ax[0].set_ylabel('loss', bbox=box)

    fig.savefig(savepath, bbox_inches='tight', dpi=200, transparent=True)


if __name__ == '__main__':

    args = parse_args()

    summarize(
        args.path,
        args.cp_dir,
        args.infer_cp_dir,
        args.train_metric,
        args.eval_metric,
        args.overwrite)
