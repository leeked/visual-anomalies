import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import os
import yaml
import argparse
from models import get_model
from utils.dataset import ObjectDetectionDataset
from utils.transforms import get_transform
from utils.sampler import BalancedSampler  # Ensure this is imported if using class imbalance handling

def main(config):
    # Set seeds for reproducibility
    seed = config['training']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    split_ratios = (
        config['data']['train_split'],
        config['data']['val_split'],
        config['data']['test_split']
    )

    datasets = {
        'train': ObjectDetectionDataset(
            data_dir=config['data']['data_dir'],
            split='train',
            transforms=get_transform(train=True, config=config),
            split_ratios=split_ratios,
            seed=seed
        ),
        'val': ObjectDetectionDataset(
            data_dir=config['data']['data_dir'],
            split='val',
            transforms=get_transform(train=False, config=config),
            split_ratios=split_ratios,
            seed=seed
        )
    }

    num_classes = len(datasets['train'].get_class_names())
    config['model']['num_classes'] = num_classes

    model = get_model(config, num_classes)
    model = model.to(device)

    # Determine if we need to use BalancedSampler
    imbalance_method = config['training'].get('class_imbalance_handling', {}).get('method', 'none')
    if imbalance_method == 'balanced_sampler':
        train_sampler = BalancedSampler(
            datasets['train'],
            num_samples=len(datasets['train']),
            replacement=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=config['training']['batch_size'],
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=config['data']['num_workers'],
            collate_fn=lambda x: tuple(zip(*x))
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            collate_fn=lambda x: tuple(zip(*x))
        )
    }

    weight_decay = config['training'].get('weight_decay', 0.0)

    optimizer = None
    if config['training']['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=weight_decay)
    elif config['training']['optimizer'] == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer type")

    scheduler = None
    scheduler_name = config['training']['scheduler']['name']
    if scheduler_name == 'step_lr':
        scheduler = StepLR(
            optimizer,
            step_size=config['training']['scheduler']['step_size'],
            gamma=config['training']['scheduler']['gamma']
        )
    elif scheduler_name == 'cosine_annealing':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['scheduler']['T_max'],
            eta_min=config['training']['scheduler'].get('eta_min', 0)
        )
    elif scheduler_name == 'cosine_annealing_warm_restarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['training']['scheduler']['T_0'],
            T_mult=config['training']['scheduler'].get('T_mult', 1),
            eta_min=config['training']['scheduler'].get('eta_min', 0)
        )
    else:
        raise ValueError("Unsupported scheduler type")

    num_epochs = config['training']['epochs']
    best_loss = float('inf')
    early_stopping_enabled = config['training'].get('early_stopping', {}).get('enabled', False)
    early_stopping_patience = config['training'].get('early_stopping', {}).get('patience', 5)
    epochs_no_improve = 0

    # Initialize the GradScaler for mixed precision
    use_amp = config['training'].get('use_amp', False)
    if use_amp:
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                torch.set_grad_enabled(True)
            else:
                model.train()  # Keep model in train mode to get losses
                torch.set_grad_enabled(False)

            running_loss = 0.0

            for batch_idx, (images, targets) in enumerate(dataloaders[phase]):
                images = list(img.to(device) for img in images)
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]

                if phase == 'train':
                    optimizer.zero_grad()

                if use_amp:
                    with torch.autocast(device_type=device.type):
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    if phase == 'train':
                        scaler.scale(losses).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    if phase == 'train':
                        losses.backward()
                        optimizer.step()

                running_loss += losses.item() * len(images)

                # Update scheduler per batch for CosineAnnealingWarmRestarts
                if phase == 'train' and scheduler_name == 'cosine_annealing_warm_restarts':
                    scheduler.step(epoch + batch_idx / len(dataloaders[phase]))

            torch.set_grad_enabled(True)  # Re-enable gradient computation

            epoch_loss = running_loss / len(datasets[phase])
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'train' and scheduler_name != 'cosine_annealing_warm_restarts':
                scheduler.step()

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if early_stopping_enabled and epochs_no_improve >= early_stopping_patience:
                    print('Early stopping triggered')
                    model.load_state_dict(best_model_wts)
                    if not os.path.exists(config['logging']['checkpoint_dir']):
                        os.makedirs(config['logging']['checkpoint_dir'])
                    torch.save(
                        model.state_dict(),
                        os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth')
                    )
                    return  # Exit training loop

    model.load_state_dict(best_model_wts)
    if not os.path.exists(config['logging']['checkpoint_dir']):
        os.makedirs(config['logging']['checkpoint_dir'])
    torch.save(
        model.state_dict(),
        os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth')
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a bounding box classification model.'
    )
    parser.add_argument(
        '--config',
        default='configs/default.yaml',
        help='Path to the config file.'
    )
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config)
