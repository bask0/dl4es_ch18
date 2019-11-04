
import torch


class Trainer(object):
    def __init__(
            self,
            train_loader,
            eval_loader,
            model,
            optimizer,
            loss_fn,
            train_seq_length,
            is_test):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.model.weight_init()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_seq_length = train_seq_length
        self.is_test = is_test

        self.num_batches_per_epoch = 2 if is_test else 10e10

        self.epoch = 0
        self.global_step = 0

    def train_epoch(self):
        self.model.train()

        total_loss = 0

        for step, (features, targets, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            t_start = torch.randint(
                0, int(features.size(1) - self.train_loader.dataset.num_warmup_steps - self.train_seq_length), (1, ))
            t_end = t_start + self.train_loader.dataset.num_warmup_steps + self.train_seq_length

            if torch.cuda.is_available():
                features = features[:, t_start:t_end, :].cuda(non_blocking=True)
                # Targets loaded to GPU during forward pass.
                targets = targets[:, t_start:t_end].cuda(non_blocking=True)

            if torch.isnan(features).any():
                raise ValueError(
                    'NaN in features in training, training stopped.')
            if torch.isnan(targets).any():
                raise ValueError(
                    'NaN in target in training, training stopped.')

            pred = self.model(features)
            loss = self.loss_fn(
                pred[:, self.train_loader.dataset.num_warmup_steps:, 0],
                targets[:, self.train_loader.dataset.num_warmup_steps:])

            loss.backward()

            if torch.isnan(loss):
                raise ValueError('Training loss is NaN, training stopped.')

            self.optimizer.step()

            total_loss += loss.item()

            self.global_step += 1

            del loss

            if step > self.num_batches_per_epoch:
                break

        mean_loss = total_loss / (step + 1)

        self.epoch += 1

        return {
            'epoch': self.epoch,
            'loss_train': mean_loss
        }

    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()

        total_loss = 0

        for step, (features, targets, _) in enumerate(self.eval_loader):

            if torch.cuda.is_available():
                features = features.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            if torch.isnan(features).any():
                raise ValueError(
                    'NaN in features in evaluation, training stopped.')
            if torch.isnan(targets).any():
                raise ValueError(
                    'NaN in targets in evaluation, training stopped.')

            pred = self.model(features)
            loss = self.loss_fn(
                pred[:, self.eval_loader.dataset.num_warmup_steps:, 0],
                targets[:, self.eval_loader.dataset.num_warmup_steps:])

            if torch.isnan(loss):
                raise ValueError('Eval loss is NaN, training stopped.')

            total_loss += loss.item()

            del loss

            if step > self.num_batches_per_epoch:
                break

        mean_loss = total_loss / (step + 1)

        return {
            'loss_eval': mean_loss
        }

    @torch.no_grad()
    def predict(self):
        self.model.eval()

        P = self.eval_loader.dataset.get_empty_xr()
        varname = self.eval_loader.dataset.dyn_target_name
        varname_obs = self.eval_loader.dataset.dyn_target_name + '_obs'

        total_loss = 0

        for step, (features, targets, (lat, lon)) in enumerate(self.eval_loader):

            if torch.cuda.is_available():
                features = features.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            if torch.isnan(features).any():
                raise ValueError(
                    'NaN in features in evaluation, training stopped.')
            if torch.isnan(targets).any():
                raise ValueError(
                    'NaN in targets in evaluation, training stopped.')

            pred = self.model(features)
            pred = pred[:, self.eval_loader.dataset.num_warmup_steps:, 0]
            targets = targets[:, self.eval_loader.dataset.num_warmup_steps:]
            loss = self.loss_fn(
                pred,
                targets)

            if torch.isnan(loss):
                raise ValueError('Eval loss is NaN, training stopped.')

            total_loss += loss.item()

            del loss

            P[varname].values[lat, lon] = pred
            P[varname_obs].values[lat, lon] = targets

        mean_loss = total_loss / (step + 1)

        return {
            'loss_eval': mean_loss,
            'predictions': P
        }

    def save(self, checkpoint: str) -> None:
        """Saves the model at the provided checkpoint.

        Parameters
        ----------
        checkpoint_dir
            Path to target checkpoint file.
¨
        Returns
        ----------
        checkpoint

        """
        torch.save(
            {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                #'scheduler_state_dict': self.scheduler.state_dict()
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

        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
