"""
Analyze a ray.tune run; plot training curves, show best run etc.
"""

import argparse
from typing import Dict, Any
from ray.tune.analysis import ExperimentAnalysis
import os
import shutil
import glob2
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# from experiments.hydrology.experiment_config import get_config


def parse_args() -> Dict[str, Any]:
    """Parse arguments.

    Returns
    --------
    Dict of arguments.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-s',
        '--store',
        type=str,
        help='Run directory. Either pass `experiment` and `name` or `store`.'
    )

    parser.add_argument(
        '-m',
        '--metric',
        type=str,
        default='loss_test',
        help='The metric that is minimized, default is `loss (valid)`.'
    )

    args = parser.parse_args()

    return args


def summarize(
        store,
        metric: str = 'loss_test') -> None:

    if store[-1] == '/':
        store = store[:-1]
    experiment = store.split('/')[-3]
    name = store.split('/')[-2]

    summary_dir_cp = os.path.join(
        '/workspace/bkraft/dl_chapter14/experiments', experiment, name, 'tune', 'summary')

    # Grab newest experiment state file.
    experiment_state_files = glob2.glob(
        os.path.join(store, '*/experiment_state-*.json'))

    if len(experiment_state_files) == 0:
        raise ValueError('No experiment state file found!')

    exp_file = max(experiment_state_files, key=os.path.getctime)

    print(f'\nLoading experiment state file loaded from:\n{exp_file}\n')

    summary_dir = os.path.join(os.path.split(
        os.path.dirname(exp_file))[0], 'summary')

    if os.path.isdir(summary_dir):
        shutil.rmtree(summary_dir)
    os.makedirs(summary_dir)

    exp = ExperimentAnalysis(exp_file)
    configs = exp.dataframe()
    configs['rundir'] = [os.path.join(l, 'progress.csv')
                         for l in configs['logdir']]
    runs = []
    for i, f in enumerate(configs['rundir']):
        df = pd.read_csv(f)
        df['uid'] = i
        runs.append(df)
    runs = pd.concat(runs)

    best_run_dir = exp.get_best_logdir(metric, mode='min')
    best_run_file = os.path.join(best_run_dir, 'progress.csv')
    best_run = df = pd.read_csv(best_run_file)

    print(f'Best run ID: {best_run_dir}')

    for f in ['json', 'pkl']:
        in_file = os.path.join(best_run_dir, f'params.{f}')
        out_file = os.path.join(summary_dir, f'best_params.{f}')
        copyfile(in_file, out_file)

    # Plot runs.
    plot_all(runs, metric, os.path.join(summary_dir, 'all_runs.png'))
    plot_single(best_run, metric, os.path.join(summary_dir, 'best_run.png'))

    # Use the 'best_run_id' to find
    # all directories of this run.
    # search_pattern = os.path.join(
    #     best_run_dir, f'imgs/sample_*.png')
    # imgs = glob2.glob(search_pattern)

    # if len(imgs) == 0:
    #     logging.warning(
    #         f'No images found for {search_pattern}, cannot create .gif')
    # else:
    #     imgs = sorted(imgs, key=os.path.getmtime)
    #     imgs_joined = " ".join([f"'{im}'" for im in imgs])
    #     syscall = f'convert -dispose Background -loop 0 -delay 50 {imgs_joined} {
    #         os.path.join(summary_dir, "best_run_progress.gif")}'
    #     os.system(syscall)

    bd = os.path.split(summary_dir_cp)[0]
    if os.path.isdir(bd):
        shutil.rmtree(bd)
    os.makedirs(bd)

    shutil.copytree(summary_dir, summary_dir_cp)


def plot_all(runs: pd.core.frame.DataFrame, metric: str, savepath: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey='row', gridspec_kw={'wspace': 0, 'hspace': 0})
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

    train_name = 'loss_test'
    valid_name = 'loss_train'

    runs.groupby(['uid']).plot(
        x='epoch', y=train_name, ax=ax[0], legend=False)
    runs.groupby(['uid']).plot(
        x='epoch', y=valid_name, ax=ax[1], legend=False)

    ymin = np.min((
        np.min(runs[train_name]),
        np.min(runs[valid_name]))) * 0.98
    ymax = np.max(
        (np.percentile(runs[train_name], 99), np.percentile(runs[valid_name], 99)))
    xmin = np.min(runs['epoch'])-np.max(runs['epoch'])*0.01
    xmax = np.max(runs['epoch'])*1.01

    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(ymin, ymax)
    ax[0].yaxis.set_label_coords(-0.15, 0.5, transform=ax[0].transAxes)
    ax[0].set_ylabel('loss', bbox=box)

    fig.savefig(savepath, bbox_inches='tight', dpi=200, transparent=True)


def plot_single(single_run: pd.core.frame.DataFrame, metric: str, savepath: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey='row', gridspec_kw={'wspace': 0, 'hspace': 0})
    box = dict(facecolor='yellow', pad=6, alpha=0.2)

    ax[0].text(
        1.0, 1.0, 'BEST RUN', transform=ax[0].transAxes,
        horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
    ax[0].text(
        0.5, 0.95, 'TRAINING', transform=ax[0].transAxes,
        horizontalalignment='center', verticalalignment='top', bbox=box)
    ax[1].text(
        0.5, 0.95, 'TESTING', transform=ax[1].transAxes,
        horizontalalignment='center', verticalalignment='top', bbox=box)

    train_name = 'loss_test'
    valid_name = 'loss_train'

    single_run.plot(x='epoch', y=train_name, ax=ax[0], legend=False)
    single_run.plot(x='epoch', y=valid_name, ax=ax[1], legend=False)

    ymax = np.max((np.percentile(single_run[train_name], 99), np.percentile(
        single_run[valid_name], 99)))
    xmin = np.min(single_run['epoch'])-np.max(single_run['epoch'])*0.01
    xmax = np.max(single_run['epoch'])*1.01

    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(None, ymax)
    ax[0].yaxis.set_label_coords(-0.15, 0.5, transform=ax[0].transAxes)
    ax[0].set_ylabel('loss', bbox=box)

    fig.savefig(savepath, bbox_inches='tight', dpi=200, transparent=True)


if __name__ == '__main__':

    args = parse_args()

    summarize(args.store, args.metric)
