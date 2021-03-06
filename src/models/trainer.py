
import torch
import numpy as np
import xarray as xr
import datetime
import sys
import os
from torch.optim.lr_scheduler import LambdaLR
from warnings import warn

from utils.loggers import EpochLogger
from utils.lr_scheduler import cos_decay_with_warmup


class Trainer(object):
    def __init__(
            self,
            train_loader,
            eval_loader,
            model,
            optimizer,
            loss_fn,
            train_seq_length,
            train_sample_size=None,
            gradient_clipping=0.05):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model.cuda() if torch.cuda.is_available() else model
        # self.model.weight_init()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_seq_length = train_seq_length
        self.train_sample_size = 9999999999 if train_sample_size is None else train_sample_size
        self.gradient_clipping = gradient_clipping

        scheduler_lambda = cos_decay_with_warmup(warmup=5, T=200, start_val=0.00001)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=scheduler_lambda)

        self.epoch = 0

        self.patience_counter = 0
        self.best_loss = None

        self.epoch_logger = EpochLogger()

    def train_epoch(self):
        self.model.train()

        nan_counter = 0

        for step, (features_d, features_s, target, _) in enumerate(self.train_loader):

            if torch.cuda.is_available():
                features_d = features_d.cuda(non_blocking=True)
                features_s = features_s.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            #if torch.isnan(features_d).any():
            #    raise ValueError(
            #        'NaN in dynamic features during training, training stopped.')
            #if torch.isnan(features_s).any():
            #    raise ValueError(
            #        'NaN in static features during training, training stopped.')
            #if torch.isnan(target).any():
            #    raise ValueError(
            #        'NaN in target during training, training stopped.')

            pred = self.model(features_d, features_s)

            loss = self.loss_fn(
                pred[:, self.train_loader.dataset.num_warmup_steps:],
                target[:, self.train_loader.dataset.num_warmup_steps:])

            self.optimizer.zero_grad()
            loss.backward()

            if torch.isnan(loss):
                # This is a debugging feature, if NaNs occur, possible a bug or unstable
                # model.
                nan_counter += 1
                if nan_counter > 9:
                    raise ValueError(
                        'Training loss was NaN >5 times, training stopped.')
                warn(
                    f'Training loss was NaN {nan_counter} time{"" if nan_counter==1 else "s"} '
                    'in a row, stopping after >9.')

                continue

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clipping)

            self.optimizer.step()

            self.epoch_logger.log('loss', 'train', loss.item())

            del loss

            if step > self.train_sample_size:
                break

        self.epoch += 1

        self.scheduler.step()

        stats = self.epoch_logger.get_summary()

        return {
            'epoch': self.epoch,
            'lr': self.scheduler.get_last_lr()[0],
            **stats
        }

    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()

        for step, (features_d, features_s, target, _) in enumerate(self.eval_loader):

            if torch.cuda.is_available():
                features_d = features_d.cuda(non_blocking=True)
                features_s = features_s.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # if torch.isnan(features).any():
            #     raise ValueError(
            #         'NaN in features in evaluation, training stopped.')
            # if torch.isnan(targets).any():
            #     raise ValueError(
            #         'NaN in targets in evaluation, training stopped.')

            pred = self.model(features_d, features_s)

            loss = self.loss_fn(
                pred[:, self.eval_loader.dataset.num_warmup_steps:],
                target[:, self.eval_loader.dataset.num_warmup_steps:])

            if torch.isnan(loss):
                raise ValueError('Eval loss is NaN, training stopped.')

            self.epoch_logger.log('loss', 'eval', loss.item())

            if step > self.train_sample_size:
                break

        stats = self.epoch_logger.get_summary()

        perc_improved = self.early_stopping(stats['loss_eval'])

        return {
            **stats,
            'patience_counter': self.patience_counter,
            'perc_improved': perc_improved,
            'best_loss': self.best_loss
        }

    @torch.no_grad()
    def predict(self, target_dir, use_training_set=False):
        self.model.eval()

        if use_training_set:
            print('\npredicting training set\n')
        else:
            print('\npredicting test set\n')

        print('Prediction saved to: ', target_dir)
        if use_training_set:
            data_loader = self.train_loader
        else:
            data_loader = self.eval_loader

        xr_var = data_loader.dataset.get_empty_xr()

        pred_array = np.zeros(xr_var.shape, dtype=np.float32)
        obs_array = np.zeros(xr_var.shape, dtype=np.float32)

        pred_array.fill(np.nan)
        obs_array.fill(np.nan)

        for step, (features_d, features_s, target, (lat, lon)) in enumerate(data_loader):

            print_progress(
                np.min((
                    (step + 1) *
                    data_loader.batch_size, len(data_loader.dataset)
                )), len(data_loader.dataset), 'predicting')

            if torch.cuda.is_available():
                features_d = features_d.cuda(non_blocking=True)
                features_s = features_s.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            pred = self.model(features_d, features_s)
            pred = pred[:, data_loader.dataset.num_warmup_steps:]
            target = target[:, data_loader.dataset.num_warmup_steps:]

            pred = self.unstandardize_target(pred)
            target = self.unstandardize_target(target)

            loss = self.loss_fn(
                pred,
                target)

            lat = lat.numpy()
            lon = lon.numpy()

            pred_array[:, lat, lon] = pred.cpu().numpy().T
            obs_array[:, lat, lon] = target.cpu().numpy().T

            self.epoch_logger.log('loss', 'test', loss.item())

        print('\nWriting to file...')

        pred = xr.Dataset({
            'mod': xr.DataArray(pred_array, coords=[xr_var.time, xr_var.lat, xr_var.lon]),
            'obs': xr.DataArray(obs_array, coords=[xr_var.time, xr_var.lat, xr_var.lon])
        })
        pred.obs.attrs = xr_var.attrs
        pred.obs.attrs = xr_var.attrs

        pred.attrs = {
            'created': datetime.date.today().strftime('%b %d %Y'),
            'contact': 'bkraft@bgc-jena.mpg.de, sbesnard@bgc-jena.mpg.de',
            'description': 'LSTM emulation of physical process model (Koirala et al. (2017))',
            'var': xr_var.name,
            'long_name': xr_var.attrs['long_name']
        }

        pred_space_optim = pred.chunk({
            'lat': -1,
            'lon': -1,
            'time': 15
        })

        pred_time_optim = pred.chunk({
            'lat': 15,
            'lon': 15,
            'time': -1
        })

        if use_training_set:
            file_name_ending = '_trainset'
        else:
            file_name_ending = ''

        pred_space_optim.to_zarr(os.path.join(
            target_dir, f'pred_so{file_name_ending}.zarr'))
        pred_time_optim.to_zarr(os.path.join(
            target_dir, f'pred_to{file_name_ending}.zarr'))

        print('Done.')

        stats = self.epoch_logger.get_summary()

        return {
            **stats
        }

    def early_stopping(self, loss):

        if self.best_loss is not None:
            perc_improved = 100 * (1 - loss / self.best_loss)
            if perc_improved < 0.01:
                self.patience_counter += 1
            else:
                self.patience_counter = 0
            if loss < self.best_loss:
                self.best_loss = loss
        else:
            self.best_loss = loss
            self.perc_improved = perc_improved = 0

        return perc_improved

    def save(self, checkpoint: str) -> None:
        """Saves the model at the provided checkpoint.

        Parameters
        ----------
        checkpoint_dir
            Path to target checkpoint file.
??
        Returns
        ----------
        checkpoint

        """
        torch.save(
            {
                'epoch': self.epoch,
                'patience_counter': self.patience_counter,
                'best_loss': self.best_loss,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            },
            checkpoint
        )
        return checkpoint

    def restore(self, checkpoint: str) -> None:
        """Restores the model from a provided checkpoint.

        Parameters
        ----------
        filename
            Path to target checkpoint file.

        """
        checkpoint = torch.load(checkpoint)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            self.model.to_device('cuda')

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.patience_counter = checkpoint['patience_counter']
        self.best_loss = checkpoint['best_loss']

    def unstandardize_target(self, target):
        return self.train_loader.dataset.unstandardize(
                    target, self.train_loader.dataset.target_var)


def rprint(value):
    """Similar to print function but overwrites last line.

    Parameters
    ----------
    value
        Value to print.

    """

    sys.stdout.write(f'\r{value}')
    sys.stdout.flush()


def print_progress(i: int, n_total: int, prefix: str) -> None:
    """Print progress bar.

    E.g. with ``prefix`` = 'training':

    training:  97% ||||||||||||||||||||||||||||||||||||||||||||||||   |

    Parameters
    ----------
    i
        Current step.
    n_total
        Total number of steps.
    prefix
        Printed in front of progress bar, limited to 20 characters.

    """
    perc = np.floor((i + 1) / n_total * 100)
    n_print = 50

    n_done = int(np.floor(perc / 100 * n_print))
    n_to_go = n_print - n_done

    if perc != 100:
        n_to_go = n_to_go-1
        msg = f'{perc:3.0f}% |{"|"*n_done}>{" "*n_to_go}' + '|'
    else:
        msg = f'{perc:3.0f}% |{"|"*n_done}{" "*n_to_go}' + '|'

    rprint(f'{prefix:20s} ' + msg)
    if perc == 100:
        print('\n')
