from torchvision import transforms
import torch

class ToTensor(object):
    """Convert PIL Images in sample to Tensors."""

    def __call__(self, sample):
        image, bbox, label = sample['image'], sample['bbox'], sample['label']
        image = transforms.ToTensor()(image)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)
        return {'image': image, 'bbox': bbox, 'label': label}

class Resize(object):
    """Resize image and adjust bounding box accordingly."""

    def __init__(self, size):
        self.size = size  # size: (w, h)

    def __call__(self, sample):
        image, bbox, label = sample['image'], sample['bbox'], sample['label']
        w_old, h_old = image.size
        image = image.resize(self.size)
        w_new, h_new = self.size
        scale_w = w_new / w_old
        scale_h = h_new / h_old
        bbox[0] *= scale_w
        bbox[1] *= scale_h
        bbox[2] *= scale_w
        bbox[3] *= scale_h
        return {'image': image, 'bbox': bbox, 'label': label}

class RandomHorizontalFlip(object):
    """Horizontally flip the image and adjust bbox."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, bbox, label = sample['image'], sample['bbox'], sample['label']
        if torch.rand(1) < self.p:
            image = transforms.functional.hflip(image)
            w, _ = image.size
            bbox[0] = w - bbox[0] - bbox[2]  # Flip x coordinate
        return {'image': image, 'bbox': bbox, 'label': label}

