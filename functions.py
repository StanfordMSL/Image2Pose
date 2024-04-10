import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation as R
import json

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
    
def get_data(ratio:float,Ndata:int=None):
    # Paths
    images_dir = 'data/images'
    transforms_file = 'data/transforms.json'
    
    # Load frames
    with open(transforms_file, 'r') as json_file:
        frames = json.load(json_file)["frames"]

    # Step 1: Initialize model with the best available weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Some useful variables
    if Ndata is not None:
        frames = frames[:Ndata]

    Nimg = len(frames)
    Ntn = int(ratio*Nimg)
    Ntt = Nimg - Ntn

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
    
    dset = Image2PoseData(Xdata,Ydata)
    tn_dset, tt_dset = random_split(dset,[Ntn,Ntt])

    tn_lder = DataLoader(tn_dset, batch_size=64, shuffle=True, drop_last = False)
    tt_lder = DataLoader(tt_dset, batch_size=64, shuffle=True, drop_last = False)

    return tn_lder, tt_lder

def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    for epoch in range(num_epochs):
        losses = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device).float()
            targets = targets.to(device)

            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

        if epoch % 50 == 0:
            print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))

    torch.save(model.state_dict(), 'img2pose.pth')

def test_model(model, criterion, test_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets)
            
            # Update total loss and total samples
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    # Calculate average loss
    average_loss = total_loss / total_samples
    print(f'Test Loss: {average_loss:.4f}')