import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
from torch.utils.data import DataLoader
import os
import yaml
import argparse
from models import get_model
from utils.dataset import ObjectDetectionDataset
from utils.transforms import get_transform
import utils.metrics as metrics

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

    input_size = config['data']['input_size']

    # Prepare data
    split_ratios = (
        config['data']['train_split'],
        config['data']['val_split'],
        config['data']['test_split']
    )

    dataset = ObjectDetectionDataset(
        data_dir=config['data']['data_dir'],
        split='test',
        transforms=get_transform(train=False, input_size=input_size),
        split_ratios=split_ratios,
        seed=seed
    )

    num_classes = len(dataset.get_class_names())
    config['model']['num_classes'] = num_classes

    model = get_model(config, num_classes)
    model = model.to(device)
    checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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

    model.eval()

    class_names = dataset.get_class_names()

    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'mean_iou': []}

    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            # Since batch_size=1
            output = outputs[0]
            target = targets[0]
            # Get predicted boxes and labels
            pred_boxes = output['boxes'].cpu()
            pred_labels = output['labels'].cpu()
            # Get ground truth boxes and labels
            gt_boxes = target['boxes']
            gt_labels = target['labels']

            # Handle cases where there are no ground truth or predicted boxes
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                # No ground truth and no predictions
                batch_metrics = {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'mean_iou': 1.0}
            else:
                batch_metrics = metrics.compute_metrics(
                    pred_boxes=pred_boxes,
                    pred_labels=pred_labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    iou_threshold=config['metrics']['iou_threshold']
                )
            for k in all_metrics.keys():
                all_metrics[k].append(batch_metrics[k])

    avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    print('Evaluation Results:')
    for k, v in avg_metrics.items():
        print(f'{k}: {v:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a bounding box classification model.')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to the config file.')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config)
