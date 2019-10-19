
import torch
import os


class Trainer(object):
    def __init__(self, train_loader, valid_loader, model, optimizer, loss_fn):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = self.model.cuda() if torch.cuda.is_available() else model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.epoch = 0
        self.global_step = 0

    def train_epoch(self):
        self.model.train()

        total_loss = 0

        for step, (features, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            if torch.cuda.is_available():
                features = features.cuda()  # blocking
                targets = targets.cuda()

            pred = self.model(features)
            loss = self.loss_fn(pred, targets)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            self.global_step += 1

            del loss, features, targets

        mean_loss = total_loss / (step + 1)

        self.epoch += 1

        return {
            'epoch': self.epoch,
            'training_loss': mean_loss
        }

    @torch.no_grad()
    def valid_epoch(self):
        self.model.eval()
        total_loss = 0

        for step, (features, targets) in enumerate(self.valid_loader):
            self.optimizer.zero_grad()

            if torch.cuda.is_available():
                features = features.cuda()  # blocking
                targets = targets.cuda()

            pred = self.model(features)
            loss = self.loss_fn(pred, targets)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            del loss, features, targets

        mean_loss = total_loss / (step + 1)

        return {
            'valid_loss': mean_loss
        }

    def save(self, checkpoint: str) -> None:
        """Saves the model at the provided checkpoint.

        Parameters
        ----------
        checkpoint
            Path to target checkpoint file.
¨
        Returns
        ----------
        checkpoint

        """
        savefile = os.path.join(checkpoint, 'chkp.pt')
        torch.save(
            {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            },
            savefile
        )
        return savefile

    def restore(self, checkpoint: str) -> None:
        """Restores the model from a provided checkpoint.

        Parameters
        ----------
        filename
            Path to target checkpoint file.

        """
        checkpoint = torch.load(checkpoint)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to_device(self.device)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
