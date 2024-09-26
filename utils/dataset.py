import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class ObjectDetectionDataset(Dataset):
    def __init__(self, data_dir, split='train', transforms=None, split_ratios=(0.7, 0.15, 0.15), seed=42):
        """
        Args:
            data_dir (string): Root directory of the dataset.
            split (string): 'train', 'val', or 'test'.
            transforms (callable, optional): Optional transform to be applied on a sample.
            split_ratios (tuple): Ratios for train, val, and test splits.
            seed (int): Random seed for reproducibility.
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.samples = []
        self.split = split

        # Gather all samples
        all_samples = []
        categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        for category in categories:
            category_dir = os.path.join(data_dir, category)
            folders = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
            for folder in folders:
                folder_dir = os.path.join(category_dir, folder)
                bounding_box_file = os.path.join(folder_dir, 'bounding_box.json')
                if not os.path.exists(bounding_box_file):
                    continue
                with open(bounding_box_file, 'r') as f:
                    bbox_data = json.load(f)
                # Assume bbox_data contains bounding box coordinates as [x, y, width, height]

                annotated_dir = os.path.join(folder_dir, 'annotated')
                if not os.path.exists(annotated_dir):
                    continue
                for root, _, files in os.walk(annotated_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            img_path = os.path.join(root, file)
                            sample = {
                                'image_path': img_path,
                                'bbox': bbox_data,  # Adjust as per the actual structure
                                'label': category  # Use category as label
                            }
                            all_samples.append(sample)

        # Convert labels to numeric labels
        label_set = sorted(list(set([sample['label'] for sample in all_samples])))
        label_to_index = {label: idx for idx, label in enumerate(label_set)}
        for sample in all_samples:
            sample['label'] = label_to_index[sample['label']]

        # Split the data
        random.seed(seed)
        random.shuffle(all_samples)
        num_samples = len(all_samples)
        train_end = int(split_ratios[0] * num_samples)
        val_end = train_end + int(split_ratios[1] * num_samples)

        if split == 'train':
            self.samples = all_samples[:train_end]
        elif split == 'val':
            self.samples = all_samples[train_end:val_end]
        elif split == 'test':
            self.samples = all_samples[val_end:]
        else:
            raise ValueError(f"Invalid split name {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert("RGB")
        bbox = torch.tensor(sample['bbox'], dtype=torch.float32)
        label = sample['label']
        sample = {'image': image, 'bbox': bbox, 'label': label}
        if self.transforms:
            sample = self.transforms(sample)
        return sample

