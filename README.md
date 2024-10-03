# PyTorch Visual Anomaly Object Detection Project

This project provides code for training PyTorch vision-based object detection models with bounding box classification.

## Features

- Configurable via YAML configuration files.
- Supports custom models and custom backbones.
- Training, evaluation, and visualization scripts included.
- Computes metrics like Intersection over Union (IoU), accuracy, precision, and recall.
- Easy to extend and customize.
- **Supports mixed precision training for memory efficiency.**
- Advanced data preprocessing and augmentation using Albumentations.

## Directory Structure

```
project/ 
├── configs/ 
│ └── default.yaml 
├── data/ #ignored
│ ├── images/ 
│ │ ├── 0001.png 
│ │ ├── 0002.png 
│ │ └── ... 
│ ├── labels/ 
│ │ ├── 0001.txt 
│ │ ├── 0002.txt 
│ │ └── ... 
├── models/ 
│ ├── __init__.py 
│ ├── backbones.py 
│ └── custom_model.py 
├── utils/ 
│ ├── dataset.py 
│ ├── transforms.py 
│ └── metrics.py 
├── train.py 
├── evaluate.py 
├── visualize.py 
├── requirements.txt 
└── README.md
```

## Data Format

The data should be organized as follows:

- `data/images/`: Contains images named `0001.png`, `0002.png`, etc.
- `data/labels/`: Contains corresponding annotation files named `0001.txt`, `0002.txt`, etc.

Each annotation file should have the following format:

```
class_num x_corner y_corner width height 
class_num x_corner y_corner width height 
...
```

- Each line represents one object.
- `class_num`: Integer representing the class number.
- `x_corner`, `y_corner`, `width`, `height`: Floats representing the bounding box coordinates.
- If an image has no objects, the corresponding annotation file should be empty.

**Note**: The code automatically converts bounding boxes from the format `[xmin, ymin, width, height]` to `[xmin, ymin, xmax, ymax]` as required by Faster R-CNN.

## Usage

### Install dependencies

```pip install -r requirements.txt```

### Prepare Data

- Place your images in the `data/images/` directory.
- Place your annotations in the `data/labels/` directory, following the format specified above.

### Configure Training

- Modify the `configs/default.yaml` file to set your desired training parameters and model configurations.
- To enable mixed precision training, set `use_amp: true` under the `training` section.
- Configure data augmentations under the `data.augmentations` section in the YAML file.

### Data Augmentation and Preprocessing
The code uses Albumentations for advanced data augmentation techniques. You can configure the augmentations in the `configs/default.yaml` file under the `data.augmentations` section.

Example augmentation configurations:
```yaml
augmentations:
  horizontal_flip: true
  horizontal_flip_prob: 0.5
  vertical_flip: true
  vertical_flip_prob: 0.5
  rotation: true
  rotation_degrees: 15
  rotation_prob: 0.5
  color_jitter: true
  brightness: 0.2
  contrast: 0.2
  saturation: 0.2
  hue: 0.1
  color_jitter_prob: 0.5
```

### Train the Model

```python train.py --config configs/default.yaml```

### Evaluate the Model

```python evaluate.py --config configs/default.yaml```

### Visualize Model Predictions

```python visualize.py --config configs/default.yaml```

## Notes on Memory Optimization

- **Mixed Precision Training**: Enabled by setting `use_amp: true` in the configuration file. This can significantly reduce memory usage and speed up training.
- **Batch Size**: If you encounter memory issues, consider reducing the `batch_size` in the configuration file.
- **Model Complexity**: Using smaller backbones (e.g., `resnet18` instead of `resnet50`) can also help reduce memory usage.

## Custom Backbones

To use a custom backbone, create a Python file defining your backbone and specify the path in the configuration file under `model.custom_backbone_file`.

The custom backbone file should define a function `get_custom_backbone(config)` that returns a tuple `(backbone, backbone_out_channels)`.

Example `custom_backbone.py`:

```python
def get_custom_backbone(config):
    import torchvision
    backbone = torchvision.models.mobilenet_v2(pretrained=config['model']['pretrained']).features
    backbone_out_channels = 1280  # For MobileNetV2
    return backbone, backbone_out_channels
```

## Custom Models

To use a custom model, create a Python file defining your model and specify the path in the configuration file under `model.custom_model_file`.

The custom model should be a subclass of `torch.nn.Module` and return predictions compatible with the rest of the codebase.

## License

This project is licensed under the MIT License.