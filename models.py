import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class VisionPoseMLP(nn.Module):
    def __init__(self,layer_sizes:list[int],lock_resnet:bool=True):
        super(VisionPoseMLP, self).__init__()

        self.networks = nn.ModuleList(
            [ResNetReg(layer_sizes[0]-3,lock_resnet)] +
            [BasicMLP(layer_sizes)])
        
    def forward(self,x,v):
        y_img = self.networks[0](x)
        y_out = self.networks[1](torch.cat((y_img,v),-1))

        y_norm = y_out.clone()
        y_norm[:,-4:] = F.normalize(y_out[:,-4:],p=2,dim=1)

        return y_norm

class ResNetReg(nn.Module):
    def __init__(self, output_size,lock_resnet:bool=True):
        super(ResNetReg, self).__init__()

        # Instantiate the ResNet18 base
        layers = [resnet18(weights=ResNet18_Weights.DEFAULT),nn.ReLU()]

        # Lock the ResNet18 layers if need be (except for last)
        if lock_resnet:
            for param in layers[0].parameters():
                param.requires_grad = False

        # Modify the last layer to output the correct number of features
        layers[0].fc = nn.Linear(512,output_size)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
       
class BasicMLP(nn.Module):
    def __init__(self, hidden_layers):
        super(BasicMLP, self).__init__()
        layers = []
        prev_size = hidden_layers[0]
        for size in hidden_layers[1:-1]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, hidden_layers[-1]))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)