import torch
import torchvision
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models._utils import IntermediateLayerGetter

def get_backbone(backbone_name, pretrained):
    if backbone_name == 'resnet50':
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', pretrained=pretrained)
        backbone_out_channels = 256
    elif backbone_name == 'vgg16':
        backbone, backbone_out_channels = get_vgg_backbone(pretrained)
    elif backbone_name == 'vit':
        backbone, backbone_out_channels = get_vit_backbone(pretrained)
    elif backbone_name == 'simclr':
        backbone, backbone_out_channels = get_simclr_backbone(pretrained)
    else:
        raise ValueError(f"Backbone {backbone_name} not supported.")
    return backbone, backbone_out_channels

def get_vgg_backbone(pretrained):
    # Load the VGG16 model
    vgg = torchvision.models.vgg16(pretrained=pretrained)
    features = vgg.features

    # Define return layers
    return_layers = {'4': '0', '9': '1', '16': '2', '23': '3'}

    # Create the backbone using IntermediateLayerGetter
    backbone = IntermediateLayerGetter(features, return_layers=return_layers)

    # Define the number of channels for each return layer
    in_channels_list = [64, 128, 256, 512]

    # Create the FPN
    backbone = BackboneWithFPN(backbone, in_channels_list=in_channels_list, out_channels=256)

    return backbone, 256

def get_vit_backbone(pretrained):
    import torch.nn.functional as F
    from torchvision.models import vit_b_16

    class ViTBackbone(torch.nn.Module):
        def __init__(self, vit_model):
            super(ViTBackbone, self).__init__()
            self.conv_proj = vit_model.conv_proj
            self.encoder = vit_model.encoder

        def forward(self, x):
            x = self.conv_proj(x)  # Shape: [batch_size, hidden_dim, grid_size, grid_size]
            batch_size, hidden_dim, h, w = x.shape
            x = x.flatten(2).permute(2, 0, 1)  # Shape: [num_patches, batch_size, hidden_dim]
            x = self.encoder(x)  # Transformer encoder
            x = x.permute(1, 2, 0).reshape(batch_size, hidden_dim, h, w)  # Reshape back to feature map
            return {'0': x}

    vit_model = vit_b_16(pretrained=pretrained)
    backbone = ViTBackbone(vit_model)
    backbone_out_channels = vit_model.hidden_dim

    # Since ViT doesn't have multiple feature maps, we can wrap it in BackboneWithFPN with dummy layers
    return_layers = {'0': '0'}
    backbone = BackboneWithFPN(backbone, return_layers=return_layers, in_channels_list=[backbone_out_channels], out_channels=256)
    backbone_out_channels = 256

    return backbone, backbone_out_channels

def get_simclr_backbone(pretrained):
    # Placeholder for SimCLR backbone
    # Assume simclr_encoder is a ResNet50 trained with SimCLR
    simclr_encoder = torchvision.models.resnet50(pretrained=False)
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
    backbone = BackboneWithFPN(backbone, in_channels_list=in_channels_list, out_channels=256)
    backbone_out_channels = 256
    return backbone, backbone_out_channels

