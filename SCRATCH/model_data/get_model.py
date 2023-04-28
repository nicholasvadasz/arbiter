import torch

from torch import nn
from model_data.utils import get_classes, get_anchors, create_model
from torchvision import models
from recap import URI, CfgNode as CN


class ResNet(nn.Module):
    """ResNet model.
    """
    input_size = 100, 100
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, 2)
        self.params = {
            "head": list(self.model.fc.parameters())
        }

    def forward(self, x):
        return self.model(x)

def get_yolo():
    annotation_path = 'model_data/_annotations.txt'
    classes_path = 'model_data/_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    print("-------------------CLASS NAMES-------------------")
    print(class_names)
    print("-------------------CLASS NAMES-------------------")
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    model = create_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path='model_data/trained_weights_final.h5') # make sure you know what you freeze
    
    return model


def get_occupancy():
    occupancy_model = ResNet()
    occupancy_model, occupancy_cfg = torch.jit.load('model_data/chesscog_model.tjm'), CN.load_yaml_with_base("model_data/ResNet.yaml")

    return occupancy_model, occupancy_cfg

def get_corner():
    corner_detection_cfg = CN.load_yaml_with_base(
            "model_data/corner_detection.yaml")
    
    return corner_detection_cfg