import os
import random
import argparse
import yaml
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import get_model
from utils.dataset import ObjectDetectionDataset
from utils.transforms import get_transform
from utils.evaluator import Evaluator


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

    dataset = ObjectDetectionDataset(
        data_dir=config['data']['data_dir'],
        split='test',
        transforms=get_transform(train=False, config=config),
        split_ratios=split_ratios,
        seed=seed
    )

    num_classes = len(dataset.get_class_names())
    config['model']['num_classes'] = num_classes

    model = get_model(config, num_classes).to(device)
    checkpoint_path = os.path.join(
        config['logging']['checkpoint_dir'], 'best_model.pth'
    )

    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=lambda x: tuple(zip(*x))
    )

    evaluator = Evaluator(
        model=model,
        device=device,
        dataloader=dataloader,
        dataset=dataset,
        config=config
    )

    evaluator.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a bounding box classification model.'
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
