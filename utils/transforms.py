import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(train, config):
    augmentations = []
    if train:
        aug_cfg = config['data'].get('augmentations', {})
        if aug_cfg.get('horizontal_flip', False):
            augmentations.append(A.HorizontalFlip(p=aug_cfg.get('horizontal_flip_prob', 0.5)))
        if aug_cfg.get('vertical_flip', False):
            augmentations.append(A.VerticalFlip(p=aug_cfg.get('vertical_flip_prob', 0.5)))
        if aug_cfg.get('rotation', False):
            degrees = aug_cfg.get('rotation_degrees', 15)
            augmentations.append(A.Rotate(limit=degrees, p=aug_cfg.get('rotation_prob', 0.5)))
        if aug_cfg.get('color_jitter', False):
            brightness = aug_cfg.get('brightness', 0)
            contrast = aug_cfg.get('contrast', 0)
            saturation = aug_cfg.get('saturation', 0)
            hue = aug_cfg.get('hue', 0)
            augmentations.append(A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=aug_cfg.get('color_jitter_prob', 0.5)))
        if aug_cfg.get('random_crop', False):
            height = aug_cfg.get('crop_height', 512)
            width = aug_cfg.get('crop_width', 512)
            augmentations.append(A.RandomCrop(height=height, width=width, p=aug_cfg.get('random_crop_prob', 0.5)))
    # Normalize and convert to tensor
    normalize_mean = config['data'].get('normalize_mean', [0.485, 0.456, 0.406])
    normalize_std = config['data'].get('normalize_std', [0.229, 0.224, 0.225])
    augmentations.extend([
        A.Normalize(mean=normalize_mean, std=normalize_std),
        ToTensorV2()
    ])
    return A.Compose(augmentations, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))
