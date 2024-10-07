import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


class Visualizer:
    def __init__(self, model, device, dataloader,
                 dataset_original, config):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.dataset_original = dataset_original
        self.config = config

        self.class_names = dataset_original.get_class_names()
        self.save_dir = os.path.join(
            config['logging']['log_dir'], 'visualizations'
        )
        os.makedirs(self.save_dir, exist_ok=True)

    def visualize(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (images, targets) in enumerate(self.dataloader):
                if idx >= 5:  # Visualize 5 samples
                    break
                image = images[0].to(self.device)
                target = targets[0]

                # Get original image and target
                original_image, original_target = self.dataset_original[idx]
                img = original_image.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)

                fig, ax = plt.subplots(1)
                ax.imshow(img)

                # Draw ground truth bboxes
                self._draw_boxes(
                    ax, original_target['boxes'].numpy(),
                    original_target['labels'].numpy(), 'g', 'GT'
                )

                # Get model predictions
                outputs = self.model([image])
                pred_boxes = outputs[0]['boxes'].cpu().numpy()
                pred_labels = outputs[0]['labels'].cpu().numpy()
                scores = outputs[0]['scores'].cpu().numpy()

                self._draw_boxes(
                    ax, pred_boxes, pred_labels, 'r', 'Pred', scores
                )

                # Save the figure
                save_path = os.path.join(
                    self.save_dir, f"visualization_{idx+1}.png"
                )
                plt.savefig(save_path)
                plt.close(fig)

    def _draw_boxes(self, ax, boxes, labels, color, label_prefix, scores=None):
        for i in range(len(boxes)):
            bbox = boxes[i]
            xmin, ymin, xmax, ymax = bbox
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            if scores is not None:
                text = f"{label_prefix}: {self.class_names[labels[i]]} " \
                       f"({scores[i]:.2f})"
            else:
                text = f"{label_prefix}: {self.class_names[labels[i]]}"
            ax.text(
                xmin, ymin - 10, text,
                color=color, fontsize=12
            )
