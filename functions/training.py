import torch
import numpy as np
from models import *
from torch.utils.data import DataLoader

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def train_model(model:VisionPoseMLP, train_loader:DataLoader,
                model_name:str, useNeRF:bool=True, Neps:int=100,
                Nprt:int=50):
    # Training Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters())

    # Set the model to training mode
    model = model.to(device)
    model.train()

    for epoch in range(Neps):
        losses = []

        for inputs_xr,inputs_xn,inputs_v,targets in train_loader:
            if useNeRF is True:
                inputs_x = inputs_xn
            else:
                inputs_x = inputs_xr

            # Move data to the device
            inputs_x,inputs_v,targets = inputs_x.to(device),inputs_v.to(device),targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs_x,inputs_v)

            # Calculate the loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            # Append the loss
            losses.append(loss.item())

        if (epoch+1) % Nprt == 0:
            print("Epoch: {:3d} | Loss: {:4.5f}".format(epoch+1, sum(losses)/len(losses)))

    torch.save(model.state_dict(), 'models/'+model_name+'.pth')

def test_model(model,test_loader,useNeRF:bool=False,Nexp:int=3):
    # Testing Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss(reduction='mean')

    # Set the model to evaluation mode
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs_xr,inputs_xn,inputs_v, targets in test_loader:
            if useNeRF is True:
                inputs_x = inputs_xn
            else:
                inputs_x = inputs_xr
                
            # Move data to the device
            inputs_x,inputs_v,targets = inputs_x.to(device),inputs_v.to(device),targets.to(device)

            # Forward pass
            outputs = model(inputs_x,inputs_v)

            # Calculate the loss
            loss = criterion(outputs, targets)
            
            # Update total loss and total samples
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

    # Calculate average loss
    average_loss = total_loss / total_samples
    print("-------------------------------------------------------------")
    print(f'Test Loss: {average_loss:.4f}')
    
    # Print examples
    print('Examples:')
    for _ in range(Nexp):
        print("-------------------------------------------------------------")
        idx = np.random.randint(0,targets.size(0))
        output = outputs[idx].cpu().numpy()
        target = targets[idx].cpu().numpy()
        print(f'Output: {output}')
        print(f'Target: {target}')
        print("-------------------------------------------------------------")

def load_model(model_name):
    model_path = 'models/'+model_name+'.pth'
    model = torch.load(model_path)
    return model