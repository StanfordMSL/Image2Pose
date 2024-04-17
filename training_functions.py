import torch

def train_model(model, criterion, optimizer, train_loader, model_name, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    for epoch in range(num_epochs):
        losses = []

        for inputs, targets in train_loader:
            # Move data to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            # Append the loss
            losses.append(loss.item())

        if epoch % 50 == 0:
            print("Epoch: {:3d} | Loss: {:4.2f}".format(epoch, sum(losses)/len(losses)))

    print("Epoch: {:3d} | Loss: {:4.2f}".format(epoch, sum(losses)/len(losses)))
    torch.save(model.state_dict(), 'models/'+model_name+'.pth')

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