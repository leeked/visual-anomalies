import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        num_classes = config['model']['num_classes']
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        input_size = config['data']['input_size'] // 4  # Due to pooling
        self.classifier = nn.Linear(32 * input_size * input_size, num_classes)
        self.regressor = nn.Linear(32 * input_size * input_size, 4)  # For bounding box regression

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        class_out = self.classifier(x)
        bbox_out = self.regressor(x)
        return class_out, bbox_out

