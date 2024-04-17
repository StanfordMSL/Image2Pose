import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights

class BasicMLP(nn.Module):
    def __init__(self,layer_sizes:list[int]):
        super(BasicMLP, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        x_norm = x.clone()
        x_norm[:,-4:] = F.normalize(x[:,-4:],p=2,dim=1)

        return x_norm

class SimpleMLP(nn.Module):
    def __init__(self,layer_sizes:list[int]):
        super(SimpleMLP, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x
    
class StaggeredMLP(nn.Module):
    def __init__(self,pos_layer_sizes:list[int],att_layer_sizes:list[int]):
        super(StaggeredMLP, self).__init__()

        self.networks = nn.ModuleList(
            [SimpleMLP(pos_layer_sizes)] +
            [SimpleMLP(att_layer_sizes)])
        

    def forward(self, x):
        pos = self.networks[0](x)
        att = self.networks[1](torch.cat((x,pos),dim=1))

        att_norm = att.clone()
        att_norm = F.normalize(att,p=2,dim=1)

        x_norm = torch.cat((pos,att_norm),dim=1)

        return x_norm
    
class ExtendedMLP_v1(nn.Module):
    def __init__(self,layer_sizes:list[int]):
        super(ExtendedMLP_v1, self).__init__()

        layers = [resnet18(weights=ResNet18_Weights.DEFAULT)]
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        x_norm = x.clone()
        x_norm[:,-4:] = F.normalize(x[:,-4:],p=2,dim=1)

        return x_norm

class ExtendedMLP_v2(nn.Module):
    def __init__(self,layer_sizes:list[int]):
        super(ExtendedMLP_v2, self).__init__()

        layers = [resnet18(weights=ResNet18_Weights.DEFAULT),nn.ReLU()]
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        x_norm = x.clone()
        x_norm[:,-4:] = F.normalize(x[:,-4:],p=2,dim=1)

        return x_norm
    
class SuperExtendedMLP(nn.Module):
    def __init__(self,layer_sizes:list[int]):
        super(SuperExtendedMLP, self).__init__()

        layers = [resnet34(weights=ResNet34_Weights.DEFAULT)]
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        x_norm = x.clone()
        x_norm[:,-4:] = F.normalize(x[:,-4:],p=2,dim=1)

        return x_norm