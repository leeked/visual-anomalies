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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        transforms=get_transform(train=False),
        split_ratios=split_ratios,
        seed=seed
    )

    num_classes = len(dataset.get_class_names())
    config['model']['num_classes'] = num_classes

    model = get_model(config, num_classes)
    model = model.to(device)
    checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
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

    # Create directory to save visuals
    save_dir = os.path.join(config['logging']['log_dir'], 'visualizations')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            if idx >= 5:  # Visualize 5 samples
                break
            image = images[0].to(device)
            target = targets[0]
            img = images[0].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw ground truth bboxes
            true_boxes = target['boxes'].cpu().numpy()
            true_labels = target['labels'].cpu().numpy()
            for i in range(len(true_boxes)):
                bbox = true_boxes[i]
                xmin, ymin, xmax, ymax = bbox
                width = xmax - xmin
                height = ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, height,
                                         linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin - 10, f"GT: {class_names[true_labels[i]]}",
                        color='g', fontsize=12)

            # Get model predictions
            images_list = [image]
            outputs = model(images_list)
            pred_boxes = outputs[0]['boxes'].cpu().numpy()
            pred_labels = outputs[0]['labels'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()

            for i in range(len(pred_boxes)):
                bbox = pred_boxes[i]
                xmin, ymin, xmax, ymax = bbox
                width = xmax - xmin
                height = ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, height,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin - 10, f"Pred: {class_names[pred_labels[i]]} ({scores[i]:.2f})",
                        color='r', fontsize=12)

            # Save the figure
            save_path = os.path.join(save_dir, f"visualization_{idx+1}.png")
            plt.savefig(save_path)
            plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model predictions on test data.')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to the config file.')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config)
