import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.nn as nn
import os
import yaml
import argparse
from models.init import get_model
from utils.dataset import ObjectDetectionDataset
from utils.transforms import get_transform

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
            transforms=get_transform(train=True),
            split_ratios=split_ratios,
            seed=seed
        ),
        'val': ObjectDetectionDataset(
            data_dir=config['data']['data_dir'],
            split='val',
            transforms=get_transform(train=False),
            split_ratios=split_ratios,
            seed=seed
        )
    }

    num_classes = len(datasets['train'].get_class_names())
    config['model']['num_classes'] = num_classes

    model = get_model(config, num_classes)
    model = model.to(device)

    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=config['training']['batch_size'],
            shuffle=True,
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

    optimizer = None
    if config['training']['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    elif config['training']['optimizer'] == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
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
    else:
        raise ValueError("Unsupported scheduler type")

    num_epochs = config['training']['epochs']
    best_loss = float('inf')

    # Initialize the GradScaler for mixed precision
    use_amp = config['training'].get('use_amp', False)
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            # Keep the model in training mode during validation to get losses
            model.train()
            running_loss = 0.0
            if phase == 'train':
                pass  # Model is already in training mode
            else:
                # Ensure gradients are not computed during validation
                torch.set_grad_enabled(False)

            for images, targets in dataloaders[phase]:
                images = list(img.to(device) for img in images)
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]

                if phase == 'train':
                    optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast('cuda'):
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

            if phase == 'val':
                # Re-enable gradient computation after validation
                torch.set_grad_enabled(True)

            epoch_loss = running_loss / len(datasets[phase])
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

        if scheduler_name == 'step_lr':
            scheduler.step()
        elif scheduler_name == 'cosine_annealing':
            scheduler.step()

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
