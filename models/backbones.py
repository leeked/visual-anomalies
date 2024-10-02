# SSL Certificate Verification error workaround
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torchvision
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models._utils import IntermediateLayerGetter

def get_backbone(backbone_name, pretrained):
    if backbone_name == 'resnet50':
        # Get weights for the backbone
        if pretrained:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        else:
            weights = None
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name='resnet50', weights=weights
        )
        backbone_out_channels = 256
    elif backbone_name == 'vgg16':
        backbone, backbone_out_channels = get_vgg_backbone(pretrained)
    elif backbone_name == 'simclr':
        backbone, backbone_out_channels = get_simclr_backbone(pretrained)
    else:
        raise ValueError(f"Backbone {backbone_name} not supported.")
    return backbone, backbone_out_channels

def get_vgg_backbone(pretrained):
    # Load the VGG16 model
    if pretrained:
        weights = torchvision.models.VGG16_Weights.DEFAULT
    else:
        weights = None
    vgg = torchvision.models.vgg16(weights=weights)
    features = vgg.features

    # Define return layers
    return_layers = {'4': '0', '9': '1', '16': '2', '23': '3'}

    # Create the backbone using IntermediateLayerGetter
    backbone = IntermediateLayerGetter(features, return_layers=return_layers)

    # Define the number of channels for each return layer
    in_channels_list = [64, 128, 256, 512]

    # Create the FPN
    backbone = BackboneWithFPN(
        backbone,
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=None,
    )

    backbone_out_channels = 256
    return backbone, backbone_out_channels

def get_simclr_backbone(pretrained):
    # Placeholder for SimCLR backbone
    # Assume simclr_encoder is a ResNet50 trained with SimCLR
    simclr_encoder = torchvision.models.resnet50()
    if pretrained:
        # Load SimCLR pre-trained weights
        simclr_weights = torch.load('path_to_simclr_weights.pth')
        simclr_encoder.load_state_dict(simclr_weights)
    # Remove the fully connected layer
    modules = list(simclr_encoder.children())[:-2]
    backbone = torch.nn.Sequential(*modules)
    # Create an FPN backbone
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    in_channels_list = [256, 512, 1024, 2048]
    backbone = BackboneWithFPN(
        backbone,
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=None,
    )
    backbone_out_channels = 256
    return backbone, backbone_out_channels
