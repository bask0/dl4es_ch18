import argparse
import os
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

import ray
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

from models.lstm import LSTM
from models.trainer import Trainer
from data.data_loader import Data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--experiment',
        type=str,
        help='Configuration file (.json).',
        required=True
    )

    parser.add_argument(
        '--dl_config_file',
        type=str,
        help='Data loader configuration file.',
        default='../data/data_loader_config.json'
    )

    args, _ = parser.parse_known_args()

    return args


class Emulator(ray.tune.Trainable):
    def _setup(self, config):

        self.config = config

        model = LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['output_size'],
            dropout=config['dropout']
        )

        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(config['lr'])
        else:
            raise ValueError(f'Optimizer {config["optimizer"]} not defined.')

        if config['loss_fn'] == 'MSE':
            loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f'Loss function {config["loss_fn"]} not defined.')

        self.trainer = Trainer(
            train_loader=None,
            valid_loader=None,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn
        )

    def _train(self):
        train_stats = self.trainer.train_epoch()
        valid_stats = self.trainer.valid_epoch()

        stats = {**train_stats, **valid_stats}

        return stats

    def _save(self, path):
        # TODO: Path definition.
        return self.trainer.save(path)

    def _restore(self, path):
        self.trainer.restore(path)


def get_dataloader(config_file, partition_set, **kwargs):
    dataset = Data(config_file=config_file, partition_set=partition_set)
    dataloader = DataLoader(
        dataset=dataset,
        **kwargs
    )
    return dataloader


def tune(args):

    ngpu = torch.cuda.device_count()
    ncpu = os.cpu_count()

    max_concurrent = int(
        np.min((
            np.floor(ncpu / config['ncpu']),
            np.floor(ngpu / config['ngpu'])
        ))
    )

    print(
        '\nTuning hyperparameters;\n'
        f'  Available resources: {ngpu} GPUs | {ncpu} CPUs\n'
        f'  Number of concurrent runs: {max_concurrent}.'
    )

    bobh_search = TuneBOHB(
        space=space,
        max_concurrent=max_concurrent,
        metric=config['metric'],
        mode='min'
    )

    bohb_scheduler = HyperBandForBOHB(
        time_attr='epoch',
        metric=config['metric'],
        mode='min',
        max_t=2 if args.test else config['max_t'],
        reduction_factor=config['reduction_factor'])

    ray.tune.run(
        Emulator,
        resources_per_trial={
            'cpu': config['ncpu'],
            'gpu': config['ngpu']},
        num_samples=config['num_samples'],
        local_dir=store_train,
        raise_on_failed_trial=True,
        verbose=1,
        with_server=False,
        ray_auto_init=True,
        search_alg=bobh_search,
        scheduler=bohb_scheduler,
        # [JsonLogger, CSVLogger, tf2_compat_logger],
        loggers=[JsonLogger, CSVLogger],
        checkpoint_at_end=True,
        reuse_actors=False,
        stop={'patience_counter': -1 if args.test else config['patience']}
    )


if __name__ == '__main__':
    ray.init(include_webui=False, object_store_memory=int(50e9))
    tune()
