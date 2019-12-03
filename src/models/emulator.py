from data.data_loader import Data
from models.trainer import Trainer
from models.lstm import LSTM
from models.modules import BaseModule
from ray import tune
from torch.utils.data.dataloader import DataLoader
import torch
import os


def get_target_path(config, args, mode):
    if mode not in ['hptune', 'modeltune', 'inference']:
        raise ValueError(
            'Argument `mode` must be one of (`hptune`, `modeltune`, `inference`) '
            f'but is {mode}.')
    path = os.path.join(
        config["store"],
        config['target_var'],
        'perm' if args.permute else 'noperm',
        mode)
    return path


class Emulator(tune.Trainable):
    def _setup(self, config):

        self.config = config

        self.hc_config = config['hc_config']
        self.is_tune = self.hc_config['is_tune']

        train_loader = get_dataloader(
            self.hc_config,
            partition_set='train',
            is_tune=self.is_tune,
            small_aoi=self.hc_config['small_aoi'],
            fold=-1,
            permute=self.hc_config['permute'],
            batch_size=self.hc_config['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=self.hc_config['num_workers'],
            pin_memory=self.hc_config['pin_memory']
        )
        eval_loader = get_dataloader(
            self.hc_config,
            partition_set='eval',
            is_tune=self.is_tune,
            small_aoi=self.hc_config['small_aoi'],
            fold=-1,
            batch_size=self.hc_config['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=self.hc_config['num_workers'],
            pin_memory=self.hc_config['pin_memory']
        )

        model = LSTM(
            input_size=train_loader.dataset.num_inputs,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=1,
            dropout_in=config['dropout_in'],
            dropout_lstm=config['dropout_lstm'],
            dropout_linear=config['dropout_linear']
        )

        if not isinstance(model, BaseModule):
            raise ValueError(
                'The model is not a subclass of models.modules:BaseModule')

        if self.hc_config['optimizer'] == 'Adam':
            optimizer = torch.optim.AdamW(
                model.parameters(), config['learning_rate'], weight_decay=config['weight_decay'])
        else:
            raise ValueError(
                f'Optimizer {self.hc_config["optimizer"]} not defined.')

        if self.hc_config['loss_fn'] == 'MSE':
            loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(
                f'Loss function {self.hc_config["loss_fn"]} not defined.')

        self.trainer = Trainer(
            train_loader=train_loader,
            eval_loader=eval_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_seq_length=self.hc_config['train_slice_length'],
            train_sample_size=self.hc_config['train_sample_size']
        )

    def _train(self):
        train_stats = self.trainer.train_epoch()
        eval_stats = self.trainer.eval_epoch()

        stats = {**train_stats, **eval_stats}

        # Disable early stopping before 'grace period' is reched.
        if stats['epoch'] < self.hc_config['grace_period']:
            stats['patience_counter'] = -1

        return stats

    def _stop(self):
        if not self.is_tune:
            self._save(os.path.dirname(os.path.dirname(self.logdir)))

    def _predict(self, prediction_file):

        self.trainer.predict(
            prediction_file)

    def _save(self, path):
        path = os.path.join(path, 'model.pth')
        return self.trainer.save(path)

    def _restore(self, path):
        self.trainer.restore(path)


def get_dataloader(
        config,
        partition_set,
        is_tune,
        small_aoi=False,
        fold=None,
        permute=False,
        **kwargs):

    dataset = Data(
        config=config,
        partition_set=partition_set,
        is_tune=is_tune,
        small_aoi=small_aoi,
        fold=fold,
        permute=permute)
    dataloader = DataLoader(
        dataset=dataset,
        **kwargs
    )
    return dataloader
