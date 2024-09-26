import torch
import torchvision
from .backbones import get_backbone

def get_model(config):
    model = None
    if config['model']['detection_model']:
        # Use an out-of-the-box object detection model
        detection_model_name = config['model']['detection_model']
        num_classes = config['model']['num_classes']
        pretrained = config['model']['pretrained']

        if hasattr(torchvision.models.detection, detection_model_name):
            model = getattr(torchvision.models.detection, detection_model_name)(pretrained=pretrained)
            # Modify the model's head to have the correct number of classes
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            if detection_model_name.startswith('fasterrcnn'):
                model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            else:
                # Handle other detection models as needed
                pass
        else:
            raise ValueError(f"Detection model {detection_model_name} not found in torchvision.models.detection")
    else:
        # Use a backbone model with custom detection head
        if config['model']['custom_model_file']:
            # Load custom model from the specified file
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_model", config['model']['custom_model_file'])
            custom_model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_model_module)
            model = custom_model_module.CustomModel(config)
        else:
            # Load backbone model
            backbone_name = config['model']['backbone']
            num_classes = config['model']['num_classes']
            pretrained = config['model']['pretrained']

            # Get the backbone model
            backbone, backbone_out_channels = get_backbone(backbone_name, pretrained)

            model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes)
    return model