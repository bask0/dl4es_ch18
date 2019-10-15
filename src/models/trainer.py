
import torch


class Trainer(object):
    def __init__(
            self,
            train_loader,
            valid_loader,
            test_loader,

            model,
            optimizer,
            loss_fn,
            is_test):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.model.weight_init()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.is_test = is_test

        self.epoch = 0
        self.global_step = 0

    def train_epoch(self):
        self.model.train()

        total_loss = 0

        for step, (features, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            if torch.cuda.is_available():
                features = features.cuda(non_blocking=False)
                # Targets loaded to GPU during forward pass.
                targets = targets.cuda(non_blocking=True)

            pred = self.model(features)
            loss = self.loss_fn(pred[:, :, 0], targets)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            self.global_step += 1

            del loss

            if self.is_test:
                if step > 2:
                    break

        mean_loss = total_loss / (step + 1)

        self.epoch += 1

        return {
            'epoch': self.epoch,
            'loss_train': mean_loss
        }

    @torch.no_grad()
    def valid_epoch(self):
        self.model.eval()

        total_loss = 0

        for step, (features, targets) in enumerate(self.valid_loader):
            self.optimizer.zero_grad()

            if torch.cuda.is_available():
                features = features.cuda(non_blocking=False)
                # Targets loaded to GPU during forward pass.
                targets = targets.cuda(non_blocking=True)

            pred = self.model(features)
            loss = self.loss_fn(pred[:, :, 0], targets)

            self.optimizer.step()

            total_loss += loss.item()

            del loss

            if self.is_test:
                if step > 2:
                    break

        mean_loss = total_loss / (step + 1)

        return {
            'loss_valid': mean_loss
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
