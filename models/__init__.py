import torch
import torchvision
from .backbones import get_backbone

def get_model(config, num_classes):
    model = None
    if config['model']['detection_model']:
        # Use an out-of-the-box object detection model
        detection_model_name = config['model']['detection_model']
        pretrained = config['model']['pretrained']

        if hasattr(torchvision.models.detection, detection_model_name):
            model_constructor = getattr(torchvision.models.detection, detection_model_name)

            if pretrained:
                weights_attr_name = model_constructor.__name__ + '_Weights'
                if hasattr(torchvision.models.detection, weights_attr_name):
                    weights_class = getattr(torchvision.models.detection, weights_attr_name)
                    weights = weights_class.DEFAULT
                else:
                    weights = None
            else:
                weights = None

            model = model_constructor(weights=weights, num_classes=num_classes)
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
            pretrained = config['model']['pretrained']

            # Check if custom backbone is provided
            if config['model']['custom_backbone_file']:
                # Load custom backbone from the specified file
                import importlib.util
                spec = importlib.util.spec_from_file_location("custom_backbone", config['model']['custom_backbone_file'])
                custom_backbone_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_backbone_module)
                backbone, backbone_out_channels = custom_backbone_module.get_custom_backbone(config)
            else:
                # Get the backbone model
                backbone, backbone_out_channels = get_backbone(backbone_name, pretrained)

            model = torchvision.models.detection.FasterRCNN(
                backbone=backbone,
                num_classes=num_classes
            )
    return model
