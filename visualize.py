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
from torchvision import transforms
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
    model = get_model(config)
    model = model.to(device)
    checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    data_transforms = transforms.Compose([
        Resize((config['data']['input_size'], config['data']['input_size'])),
        ToTensor()
    ])

    # Use the test split for visualization
    split_ratios = (config['data']['train_split'], config['data']['val_split'], config['data']['test_split'])

    dataset = ObjectDetectionDataset(
        data_dir=config['data']['data_dir'],
        split='test',
        transforms=data_transforms,
        split_ratios=split_ratios,
        seed=seed
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config['data']['num_workers'])

    model.eval()

    class_names = dataset.get_class_names()

    # Create directory to save visuals
    save_dir = os.path.join(config['logging']['log_dir'], 'visualizations')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
            if idx >= 5:  # Visualize 5 samples
                break
            inputs = sample['image'].to(device)
            labels = sample['label'].to(device)
            bboxes = sample['bbox'].to(device)
            img = sample['image'][0].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw ground truth bbox
            true_bbox = bboxes[0].cpu().numpy()
            rect = patches.Rectangle((true_bbox[0], true_bbox[1]), true_bbox[2], true_bbox[3],
                                     linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(true_bbox[0], true_bbox[1]-10, f"GT: {class_names[labels[0].item()]}",
                    color='g', fontsize=12)

            if hasattr(model, 'roi_heads'):
                # Detection model
                outputs = model(inputs)
                pred_bboxes = outputs[0]['boxes'].cpu().numpy()
                pred_labels = outputs[0]['labels'].cpu().numpy()
                scores = outputs[0]['scores'].cpu().numpy()
                for i in range(len(pred_bboxes)):
                    pred_bbox = pred_bboxes[i]
                    pred_label = pred_labels[i]
                    score = scores[i]
                    # Draw predicted bbox
                    x1, y1, x2, y2 = pred_bbox
                    width = x2 - x1
                    height = y2 - y1
                    rect = patches.Rectangle((x1, y1), width, height,
                                             linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1-10, f"Pred: {class_names[pred_label]} ({score:.2f})",
                            color='r', fontsize=12)
            else:
                outputs_class, outputs_bbox = model(inputs)
                _, preds = torch.max(outputs_class, 1)
                pred_bboxes = outputs_bbox.cpu().numpy()
                pred_labels = preds.cpu().numpy()
                # Draw predicted bbox
                pred_bbox = pred_bboxes[0]
                pred_label = pred_labels[0]
                rect = patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3],
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(pred_bbox[0], pred_bbox[1]-10, f"Pred: {class_names[pred_label]}",
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

