import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

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
                        bbox = list(map(float, parts[1:5]))  # x_corner y_corner width height
                        objects.append({'class_num': class_num, 'bbox': bbox})
            else:
                # No label file, treat as image with no objects
                pass

            sample = {
                'image_path': image_path,
                'objects': objects  # List of {'class_num': int, 'bbox': [x, y, width, height]}
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
        width, height = image.size

        # Prepare target
        targets = {}
        boxes = []
        labels = []

        for obj in sample['objects']:
            class_num = obj['class_num']
            label = self.class_num_to_index[class_num]
            bbox = obj['bbox']
            boxes.append(bbox)
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)  # [num_objects, 4]
        labels = torch.tensor(labels, dtype=torch.int64)   # [num_objects]

        targets['boxes'] = boxes
        targets['labels'] = labels

        if self.transforms:
            image = self.transforms(image)

        return image, targets

    def get_class_names(self):
        return [str(self.index_to_class_num[i]) for i in range(len(self.index_to_class_num))]
