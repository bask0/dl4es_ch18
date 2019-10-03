
import torch
from ray import tune
import argparse

from models.lstm import LSTM
from models.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--experiment',
        type=str,
        help='Configuration file (.json).',
        required=True
    )

    args, _ = parser.parse_known_args()

    return args


class Emulator(tune.Trainable):
    def _setup(self, config):

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

    def get_dataloader():
        pass
