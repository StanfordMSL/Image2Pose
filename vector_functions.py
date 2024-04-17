import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation as R
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation as R
import json

class VectorData(Dataset):
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

    model.fc = torch.nn.Linear(512,253)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Some useful variables
    if Ndata is not None:
        frames = frames[:Ndata]

    Nimg = len(frames)
    Ntn = int(ratio*Nimg)
    Ntt = Nimg - Ntn

    # Gather Training Data
    Xdata = np.zeros((256,Nimg))
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

        vgrav = transform[:3,:3]@np.array([0.0,0.0,1.0])

        Xdata[:,idx] = np.hstack((prediction,vgrav))
        Ydata[:,idx] = pose
    
    dset = VectorData(Xdata,Ydata)
    tn_dset, tt_dset = random_split(dset,[Ntn,Ntt])

    tn_lder = DataLoader(tn_dset, batch_size=64, shuffle=True, drop_last = False)
    tt_lder = DataLoader(tt_dset, batch_size=64, shuffle=True, drop_last = False)

    return tn_lder, tt_lder
