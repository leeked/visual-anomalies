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
from utils.metrics import compute_iou, match_predictions_to_ground_truth, compute_precision_recall_f1, compute_mean_iou
from utils.metrics import get_map_metric

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

    iou_thresholds = config['metrics'].get('iou_thresholds', [0.5])
    matching_iou_threshold = config['metrics'].get('matching_iou_threshold', 0.5)

    # Enable per-class metrics
    metric = get_map_metric(iou_thresholds=iou_thresholds, class_metrics=True)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou_list = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            # Since batch_size=1
            output = outputs[0]
            target = targets[0]

            # Prepare predictions and targets in the format expected by torchmetrics
            pred_boxes = output['boxes'].cpu()
            pred_scores = output['scores'].cpu()
            pred_labels = output['labels'].cpu()

            gt_boxes = target['boxes']
            gt_labels = target['labels']

            preds = [{
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels
            }]

            target_formatted = [{
                'boxes': gt_boxes,
                'labels': gt_labels
            }]

            metric.update(preds, target_formatted)

            # Compute TP, FP, FN, IoU for custom metrics
            tp, fp, fn, iou_list = match_predictions_to_ground_truth(
                pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=matching_iou_threshold
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_iou_list.extend(iou_list)

        final_metrics = metric.compute()
        print('Evaluation Results:')
        # Print mAP metrics
        for k, v in final_metrics.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                    print(f'{k}: {v:.4f}')
                else:
                    v_list = v.tolist()
                    v_str = ', '.join(f'{val:.4f}' for val in v_list)
                    print(f'{k}: [{v_str}]')
            else:
                print(f'{k}: {v}')

        # Compute overall Precision, Recall, F1 Score, and Mean IoU
        precision, recall, f1_score = compute_precision_recall_f1(total_tp, total_fp, total_fn)
        mean_iou = compute_mean_iou(total_iou_list)

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1_score:.4f}')
        print(f'Mean IoU: {mean_iou:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a bounding box classification model.')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to the config file.')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config)
