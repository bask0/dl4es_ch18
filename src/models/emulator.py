from data.data_loader import Data
from models.trainer import Trainer
from models.lstm import LSTM
from models.modules import BaseModule
from ray import tune
from torch.utils.data.dataloader import DataLoader
import torch
import os


class Emulator(tune.Trainable):
    def _setup(self, config):

        self.config = config

        self.hc_config = config['hc_config']
        self.is_tune = self.hc_config['is_tune']

        model = LSTM(
            input_size=len(self.hc_config['dynamic_vars']),
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=1,
            dropout=config['dropout']
        )

        if not isinstance(model, BaseModule):
            raise ValueError(
                'The model is not a subclass of models.modules:BaseModule')

        folds = get_cv_folds(config.get('fold', None))

        train_loader = get_dataloader(
            self.hc_config,
            partition_set='train',
            is_tune=self.is_tune,
            fold=folds['train'],
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
            fold=folds['train'],
            batch_size=self.hc_config['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=self.hc_config['num_workers'],
            pin_memory=self.hc_config['pin_memory']
        )

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
            is_test=self.hc_config['is_test']
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
            self._save(os.path.join(self.logdir, 'final'))

    def _predict(self):
        
        predictions =  self.trainer.predict()
        return predictions['predictions']

    def _save(self, path):
        path = os.path.join(path, 'model.pth')
        return self.trainer.save(path)

    def _restore(self, path):
        self.trainer.restore(path)


def get_cv_folds(fold):
    # No cross-validation:
    if fold == -1:
        return {'train': 1, 'eval': 1}
    # For hyperparameter tuning:
    elif fold is None:
        return {'train': None, 'eval': None}
    else:
        raise ValueError(f'Argument `fold`: invalid value: {fold}')


def get_dataloader(config, partition_set, is_tune, fold=None, **kwargs):
    dataset = Data(
        config=config,
        partition_set=partition_set,
        is_tune=is_tune,
        fold=fold)
    dataloader = DataLoader(
        dataset=dataset,
        **kwargs
    )
    return dataloader

