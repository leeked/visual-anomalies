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
from utils.transforms import ToTensor, Resize
import utils.metrics as metrics
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
    model.load_state_dict(torch.load(os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth')))

    data_transforms = transforms.Compose([
        Resize((config['data']['input_size'], config['data']['input_size'])),
        ToTensor()
    ])

    split_ratios = (config['data']['train_split'], config['data']['val_split'], config['data']['test_split'])

    dataset = ObjectDetectionDataset(
        data_dir=config['data']['data_dir'],
        split='test',
        transforms=data_transforms,
        split_ratios=split_ratios,
        seed=seed
    )

    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])

    model.eval()
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'mean_iou': []}

    with torch.no_grad():
        for samples in dataloader:
            inputs = samples['image'].to(device)
            labels = samples['label'].to(device)
            bboxes = samples['bbox'].to(device)
            if hasattr(model, 'roi_heads'):
                # Detection model
                outputs = model(inputs)
                pred_bboxes = [output['boxes'][0] for output in outputs]
                pred_labels = [output['labels'][0] for output in outputs]
            else:
                outputs_class, outputs_bbox = model(inputs)
                _, preds = torch.max(outputs_class, 1)
                pred_bboxes = outputs_bbox
                pred_labels = preds

            # Compute metrics
            batch_metrics = metrics.compute_metrics(
                outputs={'bbox': pred_bboxes, 'label': pred_labels},
                targets={'bbox': bboxes, 'label': labels},
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

