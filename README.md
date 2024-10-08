# PyTorch Visual Anomaly Object Detection Project

This project provides code for training PyTorch vision-based object detection models with bounding box classification.

Features
--------

- Configurable via YAML configuration files.
- Supports custom models and custom backbones.
- Training, evaluation, and visualization scripts included.
- Computes industry-standard object detection metrics:
  - Mean Average Precision (mAP) at different IoU thresholds.
  - Precision, Recall, F1 Score, and Mean IoU.
  - Per-class metrics for detailed analysis.
- Easy to extend and customize.
- Supports mixed precision training for memory efficiency.
- Advanced data preprocessing and augmentation using Albumentations.
- Handles class imbalance during training using balanced sampling.
- Implements overfitting mitigation techniques such as early stopping
  and weight decay.
- Supports Cosine Annealing with Warm Restarts scheduler for training.
- Integrated debugging system using Python's logging module for
  detailed tracking of training variables and issues.
- **Supports multi-GPU and multi-node distributed training using
  PyTorch's Distributed Data Parallel (DDP).**

Distributed Training
--------------------

To leverage multiple GPUs and multiple nodes for training, the codebase
supports distributed training using PyTorch's Distributed Data Parallel (DDP).

### Running Distributed Training

To run distributed training, you need to launch the training script using
`torch.distributed.launch` or `torchrun`.

#### Using `torch.distributed.launch` (PyTorch < 1.9)

```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train.py --config configs/default.yaml
```

**Using `torchrun` (PyTorch >= 1.9)**

```bash
torchrun --nproc_per_node=NUM_GPUS train.py --config configs/default.yaml
```

Replace `NUM_GPUS` with the number of GPUs you want to use per node.

For multi-node training, you need to set additional environment variables such as `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`.

Example for multi-node training:

```bash
torchrun --nproc_per_node=NUM_GPUS --nnodes=NUM_NODES --node_rank=NODE_RANK --master_addr=MASTER_ADDR --master_port=MASTER_PORT train.py --config configs/default.yaml
```

- `NUM_NODES`: Total number of nodes participating in the training.

- `NODE_RANK`: Rank of the current node (from `0` to `NUM_NODES - 1`).

- `MASTER_ADDR`: Address of the master node (e.g., IP address).

- `MASTER_PORT`: Port on the master node for communication.

**Notes**

- The code automatically initializes the distributed process group and wraps the model with `DistributedDataParallel`.

- The `DistributedSampler` is used for the training dataset to ensure that each process works on a unique subset of data.

- Logging and checkpoint saving are performed only on the main process (rank `0`).

Debugging System
----------------

To facilitate easier debugging and tracking of variables during training,
the codebase uses Python's `logging` module. The logging configuration
can be adjusted via the `configs/default.yaml` file under the `logging`
section.

Example logging configuration:

```yaml
logging:
  log_dir: logs
  checkpoint_dir: checkpoints
  log_level: 'DEBUG'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
```

- **Log Levels:** You can set the `log_level` to control the verbosity of the logs. For detailed debugging information, set it to `'DEBUG'`.
- **Log Files:** Logs are saved to `training.log` in the specified `log_dir`. Logs are also printed to the console.
- **Debug Information:** The logs include detailed information about the training process, including batch losses, epochs, model saving, and any errors that occur during training.

Data Format
-----------

The data should be organized as follows:

- `data/images/`: Contains images named `0001.png`, `0002.png`, etc.
- `data/labels/`: Contains corresponding annotation files named
  `0001.txt`, `0002.txt`, etc.

Each annotation file should have the following format:

```
class_num x_corner y_corner width height 
class_num x_corner y_corner width height 
...
```

Each line represents one object.

- `class_num`: Integer representing the class number.
- `x_corner`, `y_corner`, `width`, `height`: Floats representing the
  bounding box coordinates.

If an image has no objects, the corresponding annotation file should be
empty.

Note: The code automatically converts bounding boxes from the format
`[xmin, ymin, width, height]` to `[xmin, ymin, xmax, ymax]` as required
by Faster R-CNN.

Usage
-----

### Install dependencies

```pip install -r requirements.txt```

Some dependencies like `pycocotools` may require a C compiler to
install.

### Prepare Data

- Place your images in the `data/images/` directory.
- Place your annotations in the `data/labels/` directory, following the
  format specified above.

### Configure Training

- Modify the `configs/default.yaml` file to set your desired training
  parameters and model configurations.
- To enable mixed precision training, set `use_amp: true` under the
  `training` section.
- Configure data augmentations under the `data.augmentations` section
  in the YAML file.
- To use Cosine Annealing with Warm Restarts, set the scheduler name to
  `cosine_annealing_warm_restarts` and specify the necessary parameters.

### Train the Model

To enable detailed debugging information, set the `log_level` to `'DEBUG'` in your configuration file.

For single GPU training:

```python train.py --config configs/default.yaml```

### Evaluate the Model

The evaluation script computes industry-standard object detection metrics including Mean Average Precision (mAP), Precision, Recall, F1 Score, and Mean IoU at different IoU thresholds.

Run the evaluation script:

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