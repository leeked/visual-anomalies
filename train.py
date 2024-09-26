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
from models import get_model
from utils.dataset import ObjectDetectionDataset
from utils.transforms import ToTensor, Resize, RandomHorizontalFlip
from torchvision import transforms

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
    model = get_model(config)
    model = model.to(device)

    data_transforms = {
        'train': transforms.Compose([
            Resize((config['data']['input_size'], config['data']['input_size'])),
            RandomHorizontalFlip(),
            ToTensor()
        ]),
        'val': transforms.Compose([
            Resize((config['data']['input_size'], config['data']['input_size'])),
            ToTensor()
        ]),
    }

    split_ratios = (config['data']['train_split'], config['data']['val_split'], config['data']['test_split'])

    datasets = {
        'train': ObjectDetectionDataset(
            data_dir=config['data']['data_dir'],
            split='train',
            transforms=data_transforms['train'],
            split_ratios=split_ratios,
            seed=seed
        ),
        'val': ObjectDetectionDataset(
            data_dir=config['data']['data_dir'],
            split='val',
            transforms=data_transforms['val'],
            split_ratios=split_ratios,
            seed=seed
        )
    }

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers']),
        'val': DataLoader(datasets['val'], batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    }

    optimizer = None
    if config['training']['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    elif config['training']['optimizer'] == 'sgd':
        optimizer = SGD(model.parameters(), lr=config['training']['learning_rate'], momentum=0.9)
    elif config['training']['optimizer'] == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    else:
        raise ValueError("Unsupported optimizer type")

    scheduler = None
    scheduler_name = config['training']['scheduler']['name']
    if scheduler_name == 'step_lr':
        scheduler = StepLR(optimizer, step_size=config['training']['scheduler']['step_size'], gamma=config['training']['scheduler']['gamma'])
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

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            for samples in dataloaders[phase]:
                inputs = samples['image'].to(device)
                labels = samples['label'].to(device)
                bboxes = samples['bbox'].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if hasattr(model, 'roi_heads'):  # Detection model
                        targets = []
                        for i in range(len(labels)):
                            target = {}
                            target['boxes'] = bboxes[i].unsqueeze(0)
                            target['labels'] = labels[i].unsqueeze(0)
                            targets.append(target)
                        loss_dict = model(inputs, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    else:
                        outputs_class, outputs_bbox = model(inputs)
                        loss_class = nn.CrossEntropyLoss()(outputs_class, labels)
                        loss_bbox = nn.SmoothL1Loss()(outputs_bbox, bboxes)
                        losses = loss_class + loss_bbox

                    if phase == 'train':
                        losses.backward()
                        optimizer.step()
                running_loss += losses.item() * inputs.size(0)
            epoch_loss = running_loss / len(datasets[phase])
            print(f'{phase} Loss: {epoch_loss:.4f}')
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

        if scheduler_name == 'step_lr':
            scheduler.step()
        elif scheduler_name == 'cosine_annealing' and phase == 'train':
            scheduler.step()

    model.load_state_dict(best_model_wts)
    if not os.path.exists(config['logging']['checkpoint_dir']):
        os.makedirs(config['logging']['checkpoint_dir'])
    torch.save(model.state_dict(), os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a bounding box classification model.')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to the config file.')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config)
