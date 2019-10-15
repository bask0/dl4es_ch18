import argparse
import os
import sys
import shutil
import numpy as np
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
from torch.utils.data.dataloader import DataLoader

import ray
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.logger import CSVLogger, JsonLogger

from models.modules import BaseModule
from models.lstm import LSTM
from models.trainer import Trainer
from data.data_loader import Data
from experiments.hydrology.experiment_config import get_search_space, get_config


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
        '--predict',
        '-P',
        help='Flag to set prediction mode (hyperparameter optimization).',
        action='store_true'
    )

    parser.add_argument(
        '--overwrite',
        '-O',
        help='Flag to overwrite existing runs (all existng runs will be lost!).',
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


class Emulator(ray.tune.Trainable):
    # Hard-coded configuration; used to bypass the '_setup' method below,
    # as we dont want to pass all the hard-coded arguments as search space.
    # Needs to be set before ray.tune.run(Emulator, ...) is called, usig
    # Emulator:set_hc_config. Can this be avoided?
    # TODO: Find better way to do this.
    hc_config = None

    @classmethod
    def set_hc_config(cls, hc_config):
        cls.hc_config = hc_config

    def _setup(self, config):

        if self.hc_config is None:
            raise ValueError(
                'Set hard-coded configuration using `Emulator:set_hc_config` '
                'before initializing `Emulator`.')

        self.config = config

        model = LSTM(
            input_size=len(self.hc_config['dynamic_vars']),
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=1,
            dropout=config['dropout']
        )

        if not isinstance(model, BaseModule):
            raise ValueError('The model is not a subclass of models.modules:BaseModule')

        train_loader = get_dataloader(
            self.hc_config,
            partition_set='train',
            batch_size=self.hc_config['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=self.hc_config['num_workers'],
            pin_memory=self.hc_config['pin_memory']
        )
        valid_loader = get_dataloader(
            self.hc_config,
            partition_set='valid',
            batch_size=self.hc_config['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=self.hc_config['num_workers'],
            pin_memory=self.hc_config['pin_memory']
        )
        test_loader = get_dataloader(
            self.hc_config,
            partition_set='test',
            batch_size=self.hc_config['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=self.hc_config['num_workers'],
            pin_memory=self.hc_config['pin_memory']
        )

        if self.hc_config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), config['learning_rate'])
        else:
            raise ValueError(f'Optimizer {self.hc_config["optimizer"]} not defined.')

        if self.hc_config['loss_fn'] == 'MSE':
            loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f'Loss function {self.hc_config["loss_fn"]} not defined.')

        self.trainer = Trainer(
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            is_test=self.hc_config['is_test']
        )

    def _train(self):
        train_stats = self.trainer.train_epoch()
        valid_stats = self.trainer.valid_epoch()

        stats = {**train_stats, **valid_stats}

        # Disable early stopping before 'grace period' is reched.
        if stats['epoch'] < self.hc_config['grace_period']:
            stats['patience_counter'] = -1

        return stats

    def _stop(self):
        if self.hc_config['predict']:
            # TODO: Save pedictions.
            pass

    def _save(self, path):
        path = os.path.join(path, 'model.pth')
        return self.trainer.save(path)

    def _restore(self, path):
        self.trainer.restore(path)


def get_dataloader(config, partition_set, **kwargs):
    dataset = Data(config=config, partition_set=partition_set)
    dataloader = DataLoader(
        dataset=dataset,
        **kwargs
    )
    return dataloader


def tune(args):

    search_space = get_search_space(args.config_name)
    config = get_config(args.config_name)

    store = f'{config["store"]}/{args.config_name}/{"pred" if args.predict else "tune"}'
    if args.overwrite:
        if os.path.isdir(store):
            shutil.rmtree(store)
    else:
        if os.path.isdir(store):
            raise ValueError(
                f'The tune directory {store} exists. Set flag "--overwrite" '
                'if you want to overwrite runs - all esisting runs will be lost!')
    os.makedirs(store)

    config.update({
        'predict': args.predict,
        'store': store,
        'is_test': args.test
    })

    Emulator.set_hc_config(config)

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
        max_t=config['max_t'],
        reduction_factor=config['halving_factor'])

    if args.run_single:
        logging.warning('Starting test run.')
        e = Emulator(search_space.sample_configuration())
        logging.warning('Starting training loop.')
        e._train()
        logging.warning('Finishing test run.')
        sys.exit('0')

    if not args.predict:
        ray.tune.run(
            Emulator,
            name=config['experiment_name'],
            resources_per_trial={
                'cpu': config['ncpu_per_run'],
                'gpu': config['ngpu_per_run']},
            num_samples=config['num_samples'],
            local_dir=store,
            raise_on_failed_trial=True,
            verbose=1,
            with_server=False,
            ray_auto_init=False,
            search_alg=bobh_search,
            scheduler=bohb_scheduler,
            loggers=[JsonLogger, CSVLogger],
            checkpoint_at_end=True,
            reuse_actors=False,
            stop={'patience_counter': config['patience']}
        )
    else:
        ray.tune.run(
            Emulator,
            name=config['experiment_name'],
            resources_per_trial={
                'cpu': config['ncpu_per_run'],
                'gpu': config['ngpu_per_run']},
            num_samples=1,
            local_dir=store,
            raise_on_failed_trial=True,
            verbose=1,
            with_server=False,
            ray_auto_init=False,
            loggers=[JsonLogger, CSVLogger],
            checkpoint_at_end=True,
            reuse_actors=False,
            stop={'patience_counter': -1 if args.test else config['patience']}
        )


if __name__ == '__main__':
    args = parse_args()
    print(args)
    ray.init(include_webui=False, object_store_memory=int(50e9))
    tune(args)
