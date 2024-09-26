# PyTorch Object Detection Training Boilerplate

This project provides boilerplate code for training PyTorch vision-based object detection models with bounding box classification.

## Features

- Configurable via YAML configuration files.
- Supports custom models and pre-trained models from torchvision.
- Training and evaluation scripts included.
- Computes metrics like Intersection over Union (IoU), accuracy, precision, and recall.
- Easy to extend and customize.

## Directory Structure

```
project/ 
├── configs/ 
│ └── default.yaml 
├── models/ 
│ ├── init.py 
│ ├── backbones.py 
│ └── custom_model.py 
├── utils/ 
│ ├── init.py 
│ ├── dataset.py 
│ ├── transforms.py 
│ └── metrics.py 
├── train.py 
├── evaluate.py 
├── requirements.txt 
└── README.md
```

## Usage

### Install dependencies

```
pip install -r requirements.txt
```

### Configure Training

- Modify the `configs/default.yaml` file to set your desired training parameters and model configurations.

### Train the Model

```
python train.py --config configs/default.yaml
```

### Evaluate the Model

```
python evaluate.py --config configs/default.yaml
```

## Custom Models

- To use a custom model, create a Python file defining your model and specify the path in the configuration file under `model.custom_model_file`.
- The custom model should be a subclass of `torch.nn.Module` and return both classification and bounding box regression outputs.

## License

This project is licensed under the MIT License.