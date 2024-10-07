import os
import random
import argparse
import yaml
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
)

from models import get_model
from utils.dataset import ObjectDetectionDataset
from utils.transforms import get_transform
from utils.sampler import BalancedSampler
from utils.trainer import Trainer
import logging
from utils.logger import setup_logging

def main(config):
    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Starting training script")

    # Set seeds for reproducibility
    seed = config['training']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    logger.debug(f"Seed set to {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare data
    split_ratios = (
        config['data']['train_split'],
        config['data']['val_split'],
        config['data']['test_split']
    )
    logger.debug(f"Split ratios: {split_ratios}")

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

    num_classes = datasets['train'].num_classes
    config['model']['num_classes'] = num_classes
    logger.info(f"Number of classes: {num_classes}")

    model = get_model(config, num_classes).to(device)
    logger.info(f"Model {config['model']['detection_model']} initialized")

    # Determine if we need to use BalancedSampler
    imbalance_method = config['training'].get(
        'class_imbalance_handling', {}
    ).get('method', 'none')
    logger.debug(f"Class imbalance handling method: {imbalance_method}")

    if imbalance_method == 'balanced_sampler':
        train_sampler = BalancedSampler(
            datasets['train'],
            num_samples=len(datasets['train']),
            replacement=True
        )
        shuffle = False
        logger.info("Using BalancedSampler for training")
    else:
        train_sampler = None
        shuffle = True
        logger.info("Using default data sampler for training")

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
    logger.info("Dataloaders initialized")

    weight_decay = config['training'].get('weight_decay', 0.0)
    optimizer_type = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']
    logger.debug(f"Optimizer: {optimizer_type}, Learning rate: {learning_rate}, Weight decay: {weight_decay}")

    if optimizer_type == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError("Unsupported optimizer type")
    logger.info(f"Optimizer {optimizer_type} initialized")

    scheduler_config = config['training']['scheduler']
    scheduler_name = scheduler_config['name']
    logger.debug(f"Scheduler: {scheduler_name}")

    if scheduler_name == 'step_lr':
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )
    elif scheduler_name == 'cosine_annealing':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config.get('eta_min', 0)
        )
    elif scheduler_name == 'cosine_annealing_warm_restarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config['T_0'],
            T_mult=scheduler_config.get('T_mult', 1),
            eta_min=scheduler_config.get('eta_min', 0)
        )
    else:
        raise ValueError("Unsupported scheduler type")
    logger.info(f"Scheduler {scheduler_name} initialized")

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        datasets=datasets,
        config=config
    )

    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed")


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
