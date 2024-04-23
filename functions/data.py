from torch.utils.data import DataLoader,Dataset,random_split
from torchvision.models import ResNet18_Weights
import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation as R
import json


class VisionPoseData(Dataset):
    """
    Pytorch Dataset class for the experience data.
    """
    def __init__(self,Xr:np.ndarray,Xn:np.ndarray,V:np.ndarray,Y:np.ndarray):
        self.Xr = Xr.astype(np.float32)
        self.Xn = Xn.astype(np.float32)
        self.V = V.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.Y.shape[1]

    def __getitem__(self,idx):
        xr = self.Xr[idx,:,:,:]
        xn = self.Xn[idx,:,:,:]
        v = self.V[:,idx]
        y = self.Y[:,idx]
        
        return xr,xn,v,y
    
def get_data(ratio:float,Ndata:int=None):
    # Paths
    transforms_file = 'data/nerf_images/transforms.json'
    real_file = 'data/nerf_images'
    nerf_file = 'data/nerf_images'

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
    XXrdata = []
    XXndata = []
    VVdata = []
    YYdata = []
    for frame in frames:
        # Get the raw image and transform
        file_name = frame["file_path"].split('/')[-1]
        img_raw_real = Image.open(f'{real_file}/{file_name}')
        img_raw_nerf = Image.open(f'{nerf_file}/{file_name}')

        transform = np.array(frame["transform_matrix"])

        # Process into ResNet amenable image, gravity vector and pose vector
        img_rnet_real = preprocess(img_raw_real).unsqueeze(0).numpy()
        img_rnet_nerf = preprocess(img_raw_nerf).unsqueeze(0).numpy()

        v_grav = transform[:3,:3] @ np.array([0,0,-1])

        q_img = R.from_matrix(transform[:3,:3]).as_quat()
        t_img = transform[:3,3]
        pose = np.hstack((t_img,q_img))

        # Append to the data
        XXrdata.append(img_rnet_real)
        XXndata.append(img_rnet_nerf)
        VVdata.append(v_grav)
        YYdata.append(pose)
    
    Xrdata = np.concatenate(XXrdata,axis=0)
    Xndata = np.concatenate(XXndata,axis=0)
    Vdata = np.stack(VVdata,axis=1)
    Ydata = np.stack(YYdata,axis=1)

    dset = VisionPoseData(Xrdata,Xndata,Vdata,Ydata)
    tn_dset, tt_dset = random_split(dset,[Ntn,Ntt])

    tn_lder = DataLoader(tn_dset, batch_size=64, shuffle=True, drop_last = False)
    tt_lder = DataLoader(tt_dset, batch_size=64, shuffle=True, drop_last = False)

    return tn_lder, tt_lder

def get_resn_data(ratio:float,Ndata:int=None):
    # Paths
    transforms_file = 'data/resn_data/transforms.json'
    real_file = 'data/resn_data'
    nerf_file = 'data/resn_data'

    # Load frames
    with open(transforms_file, 'r') as json_file:
        frames = json.load(json_file)["frames"]

    # Some useful variables
    if Ndata is not None:
        frames = frames[:Ndata]

    Nimg = len(frames)
    Ntn = int(ratio*Nimg)
    Ntt = Nimg - Ntn

    # Gather Training Data
    XXrdata = []
    XXndata = []
    VVdata = []
    YYdata = []
    for frame in frames:
        # Get the raw image and transform
        file_name = frame["file_path"].split('/')[-1]
        img_raw_real = Image.open(f'{real_file}/{file_name}')
        img_raw_nerf = Image.open(f'{nerf_file}/{file_name}')

        transform = np.array(frame["transform_matrix"])

        # Process into ResNet amenable image, gravity vector and pose vector
        img_rnet_real = np.transpose(np.asarray(img_raw_real), (2, 0, 1)).reshape(1, 3, 224, 224)
        img_rnet_nerf = np.transpose(np.asarray(img_raw_nerf), (2, 0, 1)).reshape(1, 3, 224, 224)

        v_grav = transform[:3,:3] @ np.array([0,0,-1])

        q_img = R.from_matrix(transform[:3,:3]).as_quat()
        t_img = transform[:3,3]
        pose = np.hstack((t_img,q_img))

        # Append to the data
        XXrdata.append(img_rnet_real)
        XXndata.append(img_rnet_nerf)
        VVdata.append(v_grav)
        YYdata.append(pose)
    
    Xrdata = np.concatenate(XXrdata,axis=0)
    Xndata = np.concatenate(XXndata,axis=0)
    Vdata = np.stack(VVdata,axis=1)
    Ydata = np.stack(YYdata,axis=1)

    dset = VisionPoseData(Xrdata,Xndata,Vdata,Ydata)
    tn_dset, tt_dset = random_split(dset,[Ntn,Ntt])

    tn_lder = DataLoader(tn_dset, batch_size=64, shuffle=True, drop_last = False)
    tt_lder = DataLoader(tt_dset, batch_size=64, shuffle=True, drop_last = False)

    return tn_lder, tt_lder

