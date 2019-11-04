from models.emulator import Emulator
from summarize_runs import summarize
from experiments.hydrology.experiment_config import get_search_space, get_config
from ray.tune.logger import CSVLogger, JsonLogger
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
import ray
import argparse
import os
import sys
import shutil
import numpy as np
import logging
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


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
        '--dl_config_file',
        type=str,
        help='Data loader configuration file.',
        default='../data/data_loader_config.json'
    )

    parser.add_argument(
        '--overwrite',
        '-O',
        help='Flag to overwrite existing runs (all existing runs will be lost!).',
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
        logging.warning('Running experiment in test mode; Not all data is used for training!')

    return args


def tune(args):

    search_space = get_search_space(args.config_name)
    config = get_config(args.config_name)
    config.update({'is_tune': True})

    store = f'{config["store"]}/{config["experiment_name"]}/{args.config_name}/tune/'
    if args.overwrite:
        if os.path.isdir(store):
            shutil.rmtree(store)
    else:
        if os.path.isdir(store):
            raise ValueError(
                f'The tune directory {store} exists. Set flag "--overwrite" '
                'if you want to overwrite runs - all existing runs will be lost!')
    os.makedirs(store)

    config.update({
        'store': store,
        'is_test': args.test
    })

    ngpu = torch.cuda.device_count()
    ncpu = os.cpu_count()

    max_concurrent = int(
        np.min((
            np.floor(ncpu / config['ncpu_per_run']),
            np.floor(ngpu / config['ngpu_per_run'])
        ))
    )

    print(
        '\nTuning hyperparameters;\n'
        f'  Available resources: {ngpu} GPUs | {ncpu} CPUs\n'
        f'  Number of concurrent runs: {max_concurrent}\n'
    )

    bobh_search = TuneBOHB(
        space=search_space,
        max_concurrent=max_concurrent,
        metric=config['metric'],
        mode='min'
    )

    bohb_scheduler = HyperBandForBOHB(
        time_attr='epoch',
        metric=config['metric'],
        mode='min',
        max_t=5 if args.test else config['max_t'],
        reduction_factor=config['halving_factor'])

    if args.run_single:
        logging.warning('Starting test run.')
        e = Emulator(search_space.sample_configuration())
        logging.warning('Starting training loop.')
        e._train()
        logging.warning('Finishing test run.')
        sys.exit('0')

    ray.tune.run(
        Emulator,
        name=config['experiment_name'],
        config={'hc_config': config},
        resources_per_trial={
            'cpu': config['ncpu_per_run'],
            'gpu': config['ngpu_per_run']},
        num_samples=max_concurrent if args.test else config['num_samples'],
        local_dir=store,
        raise_on_failed_trial=False,
        verbose=1,
        with_server=False,
        ray_auto_init=False,
        search_alg=bobh_search,
        scheduler=bohb_scheduler,
        loggers=[JsonLogger, CSVLogger],
        keep_checkpoints_num=1,
        reuse_actors=False,
        stop={'patience_counter': config['patience']}
    )

    return store, config['metric']


if __name__ == '__main__':
    args = parse_args()

    ray.init(include_webui=False, object_store_memory=int(50e9))

    store, metric_name = tune(args)

    ray.shutdown()

    summarize(store, metric_name)
