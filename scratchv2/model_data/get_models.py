import torch

from torch import nn
from model_data.utils import get_classes, get_anchors, create_model
from torchvision import models
from recap import URI, CfgNode as CN
import yaml

INFO_PATH = 'model_data/model_info/'
WEIGHT_PATH = 'model_data/raw_models/'

def dict2obj(d):
    # checking whether object d is a
    # instance of class list
    if isinstance(d, list):
           d = [dict2obj(x) for x in d]
 
    # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(d, dict):
           return d
  
    # declaring a class
    class C:
        pass
  
    # constructor of the class passed to obj
    obj = C()
  
    for k in d:
        obj.__dict__[k] = dict2obj(d[k])
  
    return obj


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
    classes_path = INFO_PATH + '_classes.txt'
    anchors_path = INFO_PATH + 'yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416)

    model = create_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path=WEIGHT_PATH + 'trained_weights_final.h5') # make sure you know what you freeze
    
    return model


def get_occupancy():
    occupancy_model = ResNet()
    occupancy_model, occupancy_cfg = torch.jit.load(WEIGHT_PATH + 'chesscog_model.tjm'), yaml.load(open(INFO_PATH + "ResNet.yaml"), Loader=yaml.FullLoader)
    occupancy_cfg = dict2obj(occupancy_cfg)

    return occupancy_model, occupancy_cfg

def get_corner():
    corner_detection_cfg = yaml.load(open(INFO_PATH + "corner_detection.yaml"), Loader=yaml.FullLoader)
    corner_detection_cfg = dict2obj(corner_detection_cfg)

    return corner_detection_cfg