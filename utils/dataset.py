import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms.functional as F  # Added import

class ObjectDetectionDataset(Dataset):
    def __init__(self, data_dir, split='train', transforms=None, split_ratios=(0.7, 0.15, 0.15), seed=42):
        """
        Args:
            data_dir (string): Root directory of the dataset, which contains 'images/' and 'labels/' subdirectories.
            split (string): 'train', 'val', or 'test'.
            transforms (callable, optional): Optional transform to be applied on a sample.
            split_ratios (tuple): Ratios for train, val, and test splits.
            seed (int): Random seed for reproducibility.
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.samples = []
        self.split = split

        images_dir = os.path.join(data_dir, 'images')
        labels_dir = os.path.join(data_dir, 'labels')

        # Gather all samples
        all_samples = []
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Ensure consistent order

        for image_file in image_files:
            image_id = os.path.splitext(image_file)[0]
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.join(labels_dir, image_id + '.txt')

            objects = []
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() == '':
                            continue
                        parts = line.strip().split()
                        class_num = int(parts[0])
                        bbox = list(map(float, parts[1:5]))  # xmin, ymin, width, height
                        # Convert bbox from [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax]
                        xmin, ymin, width, height = bbox
                        xmax = xmin + width
                        ymax = ymin + height
                        bbox_converted = [xmin, ymin, xmax, ymax]
                        objects.append({'class_num': class_num, 'bbox': bbox_converted})
            else:
                # No label file, treat as image with no objects
                pass

            sample = {
                'image_path': image_path,
                'objects': objects  # List of {'class_num': int, 'bbox': [xmin, ymin, xmax, ymax]}
            }
            all_samples.append(sample)

        # Get the set of all class numbers
        class_nums = set()
        for sample in all_samples:
            for obj in sample['objects']:
                class_nums.add(obj['class_num'])
        class_nums = sorted(list(class_nums))
        self.class_num_to_index = {class_num: idx for idx, class_num in enumerate(class_nums)}
        self.index_to_class_num = {idx: class_num for class_num, idx in self.class_num_to_index.items()}

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

        # Prepare target
        boxes = []
        labels = []

        for obj in sample['objects']:
            class_num = obj['class_num']
            label = self.class_num_to_index[class_num]
            bbox = obj['bbox']  # [xmin, ymin, xmax, ymax]
            boxes.append(bbox)
            labels.append(label)

        if boxes:
            boxes = np.array(boxes)
            labels = np.array(labels)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=np.array(image), bboxes=target['boxes'], labels=target['labels'])
            image = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            image = F.to_tensor(image)
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)

        return image, target

    def get_class_names(self):
        return [str(self.index_to_class_num[i]) for i in range(len(self.index_to_class_num))]
