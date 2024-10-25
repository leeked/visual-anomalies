import os
import torch
import logging


class Trainer:
    def __init__(self, model, device, optimizer, scheduler,
                 dataloaders, datasets, config, train_sampler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.datasets = datasets
        self.config = config
        self.train_sampler = train_sampler

        self.num_epochs = config['training']['epochs']
        self.use_amp = config['training'].get('use_amp', False)
        self.scaler = torch.amp.GradScaler() if self.use_amp else None

        self.best_loss = float('inf')
        self.best_model_wts = None

        early_stopping = config['training'].get('early_stopping', {})
        self.early_stopping_enabled = early_stopping.get('enabled', False)
        self.early_stopping_patience = early_stopping.get('patience', 5)
        self.epochs_no_improve = 0

        self.scheduler_name = config['training']['scheduler']['name']
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        self.logger = logging.getLogger(__name__)
        self.logger.info("Trainer initialized")

        self.rank = config.get('rank', 0)
        self.world_size = config.get('world_size', 1)
        self.is_main_process = self.rank == 0

    def train(self):
        for epoch in range(self.num_epochs):
            self.logger.info(f'Epoch {epoch + 1}/{self.num_epochs}')
            self.logger.info('-' * 10)

            if self.train_sampler and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)

            for phase in ['train', 'val']:
                self._run_epoch(phase, epoch)

            if (self.early_stopping_enabled and
                self.epochs_no_improve >= self.early_stopping_patience):
                self.logger.info('Early stopping triggered')
                break

        self.save_checkpoint()
        self.logger.info('Training complete')

    def _run_epoch(self, phase, epoch):
        if phase == 'train':
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.train() # Specifically for FasterRCNN
            torch.set_grad_enabled(False)

        running_loss = 0.0
        dataloader = self.dataloaders[phase]

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(self.device) for img in images]
            targets = [
                {k: v.to(self.device) for k, v in t.items()}
                for t in targets
            ]

            if phase == 'train':
                self.optimizer.zero_grad()

                try:
                    if self.use_amp:
                        with torch.autocast(device_type=self.device.type):
                            loss_dict = self.model(images, targets)
                            losses = sum(loss for loss in loss_dict.values())
                        self.scaler.scale(losses).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss_dict = self.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        losses.backward()
                        self.optimizer.step()
                except Exception as e:
                    self.logger.error(f"Error during {phase} step: {e}")
                    continue
            else:
                with torch.no_grad():
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

            running_loss += losses.item() * len(images)

            # Log detailed information
            if self.is_main_process and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Batch {batch_idx+1}/{len(dataloader)}, "
                                  f"Loss: {losses.item():.4f}")

            # Update scheduler per batch for warm restarts
            if (phase == 'train' and
                self.scheduler_name == 'cosine_annealing_warm_restarts'):
                self.scheduler.step(
                    epoch + batch_idx / len(dataloader)
                )

        epoch_loss = running_loss / len(self.datasets[phase])
        self.logger.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')

        if (phase == 'train' and
            self.scheduler_name != 'cosine_annealing_warm_restarts'):
            self.scheduler.step()

        if phase == 'val':
            self._update_best_model(epoch_loss)

    def _update_best_model(self, epoch_loss):
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_model_wts = self.model.state_dict()
            self.epochs_no_improve = 0
            self.logger.info(f"New best model found with loss {epoch_loss:.4f}")
        else:
            self.epochs_no_improve += 1
            self.logger.debug(f"No improvement in validation loss. "
                              f"Epochs without improvement: {self.epochs_no_improve}")

    def save_checkpoint(self):
        if self.is_main_process:
            if self.best_model_wts:
                self.model.load_state_dict(self.best_model_wts)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(
                self.model.state_dict(),
                checkpoint_path
            )
            self.logger.info(f"Model saved to {checkpoint_path}")
