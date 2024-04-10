import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation as R
import json

def get_data(Ndata=None):
    # Paths
    images_dir = 'data/images'
    transforms_file = 'data/transforms.json'
    
    # Load frames
    with open(transforms_file, 'r') as json_file:
        frames = json.load(json_file)["frames"]

    if Ndata is not None:
        frames = frames[:Ndata]

    # Step 1: Initialize model with the best available weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Gather Image Data
    Nimg = len(frames)

    # Gather Training Data
    Xdata = np.zeros((1000,Nimg))
    Ydata = np.zeros((7,Nimg))
    for idx,frame in enumerate(frames):
        file_path = frame["file_path"]
        transform = np.array(frame["transform_matrix"])

        file_name = file_path.split('/')[-1]

        img = Image.open(f'{images_dir}/{file_name}')
        batch = preprocess(img).unsqueeze(0)
        prediction = model(batch).squeeze(0).detach().numpy()

        Rimg = R.from_matrix(transform[:3,:3])
        timg = transform[:3,3]

        qimg = Rimg.as_quat()

        pose = np.hstack((timg,qimg))

        Xdata[:,idx] = prediction
        Ydata[:,idx] = pose
    
    return Xdata,Ydata

class Image2PoseData(Dataset):
    """
    Pytorch Dataset class for the experience data.
    """
    def __init__(self,X:np.ndarray,Y:np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.Y.shape[1]

    def __getitem__(self,idx):
        x = self.X[:,idx]
        y = self.Y[:,idx]
        
        return x,y

def generate_dataloader(X,Y):

    # Get the total number of columns
    dset = Image2PoseData(X,Y)
    dgen = DataLoader(dset, batch_size=64, shuffle=True, drop_last = False)

    return dgen

def train_regression_model(model, criterion, optimizer, train_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    for epoch in range(num_epochs):
        losses = []
        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

        print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))

    torch.save(model.state_dict(), 'img2pose.pth')

# Define your neural network
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        return self.model(x)
