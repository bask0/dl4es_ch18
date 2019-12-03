from models.emulator import Emulator, get_target_path
from utils.summarize_runs import summarize_run
from experiments.hydrology.experiment_config import get_config
from ray.tune.logger import CSVLogger, JsonLogger
import ray
import argparse
import os
import pickle
import shutil
import numpy as np
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_name',
        '-c',
        type=str,
        help='Configuration name.',
        default='default'
    )

    parser.add_argument(
        '--overwrite',
        '-O',
        help='Flag to overwrite existing runs (all existing runs will be lost!).',
        action='store_true'
    )

    parser.add_argument(
        '--permute',
        help='Whether to permute the sequence of input of output data during training.',
        action='store_true'
    )

    parser.add_argument(
        '--test',
        '-T',
        help='Flag to perform a test run; only a fraction of the data is evaluated in each epoch.',
        action='store_true'
    )

    parser.add_argument(
        '--run_single',
        help='Bypass ray.tune and run a single model train / eval iteration.',
        action='store_true'
    )

    args = parser.parse_args()

    if args.test:
        logging.warning(
            'Running experiment in test mode; Not all data is used for training!')

    return args


def load_best_config(store):
    best_config = os.path.join(store, 'summary/best_params.pkl')
    if not os.path.isfile(best_config):
        raise ValueError(
            'Tried to load best model config, file does not exist:\n'
            f'{best_config}\nRun `summarize_results.py` to create '
            'such a file.'
        )
    with open(best_config, 'rb') as f:
        config = pickle.load(f)

    return config


def tune(args):

    config = get_config(args.config_name)
    config.update({'is_tune': False})

    tune_store = get_target_path(config, args, mode='hptune')
    store = get_target_path(config, args, mode='modeltune')

    if args.overwrite:
        if os.path.isdir(store):
            shutil.rmtree(store)
    else:
        if os.path.isdir(store):
            raise ValueError(
                f'The directory {store} exists. Set flag "--overwrite" '
                'if you want to overwrite runs - all existing runs will be lost!')
    os.makedirs(store)

    best_config = load_best_config(tune_store)
    best_config.update({'fold': -1})

    best_config.update({'hc_config': config})

    config.update({
        'store': store,
        'is_test': args.test,
        'permute': args.permute
    })

    import torch
    ngpu = torch.cuda.device_count()
    ncpu = os.cpu_count()

    max_concurrent = int(
        np.min((
            np.floor(ncpu / config['ncpu_per_run']),
            np.floor(ngpu / config['ngpu_per_run'])
        ))
    )

    print(
        '\nTuning model;\n'
        f'  Available resources: {ngpu} GPUs | {ncpu} CPUs\n'
        f'  Number of concurrent runs: {max_concurrent}\n'
    )

    ray.tune.run(
        Emulator,
        name=config['experiment_name'],
        config=best_config,
        resources_per_trial={
            'cpu': config['ncpu_per_run'],
            'gpu': config['ngpu_per_run']},
        num_samples=1,
        local_dir=store,
        raise_on_failed_trial=False,
        verbose=1,
        with_server=False,
        ray_auto_init=False,
        loggers=[JsonLogger, CSVLogger],
        keep_checkpoints_num=1,
        reuse_actors=False,
        stop={
            'patience_counter': config['patience'],
            'epoch': 5 if args.test else 99999999
        }
    )

    summarize_run(store)


if __name__ == '__main__':
    args = parse_args()

    # ray.init(include_webui=False, object_store_memory=int(50e9))
    ray.init(include_webui=False, object_store_memory=int(10e8))

    tune(args)

    ray.shutdown()
