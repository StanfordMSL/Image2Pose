import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation as R
import json


class VisionPoseData(Dataset):
    """
    Pytorch Dataset class for the experience data.
    """
    def __init__(self,X:np.ndarray,V:np.ndarray,Y:np.ndarray):
        self.X = X.astype(np.float32)
        self.V = V.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.Y.shape[1]

    def __getitem__(self,idx):
        x = self.X[idx,:,:,:]
        v = self.V[:,idx]
        y = self.Y[:,idx]
        
        return x,v,y
    
def get_data(ratio:float,Ndata:int=None):
    # Paths
    images_dir = 'data/images'
    transforms_file = 'data/transforms.json'
    
    # Load frames
    with open(transforms_file, 'r') as json_file:
        frames = json.load(json_file)["frames"]

    # Initialize the inference transforms
    preprocess = ResNet18_Weights.DEFAULT.transforms()

    # Some useful variables
    if Ndata is not None:
        frames = frames[:Ndata]

    Nimg = len(frames)
    Ntn = int(ratio*Nimg)
    Ntt = Nimg - Ntn

    # Gather Training Data
    XXdata = []
    VVdata = []
    YYdata = []
    for frame in frames:
        # Get the raw image and transform
        file_name = frame["file_path"].split('/')[-1]
        img_raw = Image.open(f'{images_dir}/{file_name}')

        transform = np.array(frame["transform_matrix"])

        # Process into ResNet amenable image, gravity vector and pose vector
        img_rnet = preprocess(img_raw).unsqueeze(0).numpy()

        v_grav = transform[:3,:3] @ np.array([0,0,-1])

        q_img = R.from_matrix(transform[:3,:3]).as_quat()
        t_img = transform[:3,3]
        pose = np.hstack((t_img,q_img))

        # Append to the data
        XXdata.append(img_rnet)
        VVdata.append(v_grav)
        YYdata.append(pose)
    
    Xdata = np.concatenate(XXdata,axis=0)
    Vdata = np.stack(VVdata,axis=1)
    Ydata = np.stack(YYdata,axis=1)

    dset = VisionPoseData(Xdata,Vdata,Ydata)
    tn_dset, tt_dset = random_split(dset,[Ntn,Ntt])

    tn_lder = DataLoader(tn_dset, batch_size=64, shuffle=True, drop_last = False)
    tt_lder = DataLoader(tt_dset, batch_size=64, shuffle=True, drop_last = False)

    return tn_lder, tt_lder

