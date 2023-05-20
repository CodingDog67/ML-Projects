import numpy as np
from datetime import datetime
import cifar_cnn_model
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# training loop batch gradient descent with accuracy computation during train time
def train(model, loss_func, optimizer, train_loader, test_loader, epochs, save_path):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    train_accs = np.zeros(epochs)
    test_accs = np.zeros(epochs)

    for it in range(epochs):
        # drop out/batch norm and others works differently in train and eval so we have to set the modes
        model.train()   
        train_loss =[]
        test_loss = []
        train_acc = []
        test_acc = []

        t_0 = datetime.now()

        n_correct = 0.
        n_total = 0.
        for inputs, targets in train_loader:
            # push data to gpu
            inputs = inputs.to(device)
            targets = targets.to(device)

            # set parameter gradients to zero 
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # Get predictions
            # torch.max returns both max and argmax
            _, predictions = torch.max(outputs, 1)
            
            # update counts
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]

            loss = loss_func(outputs, targets)

            # backward pass + optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_acc = n_correct / n_total
        train_acc = np.mean(train_acc)
        train_accs[it] = train_acc

        train_loss = np.mean(train_loss) # avg loss per epoch over all batches
        train_losses[it] = train_loss

        model.eval()
        
        n_correct = 0.
        n_total = 0.

        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs)

            # Get predictions
            # torch.max returns both max and argmax
            _, predictions = torch.max(output, 1)
            
            # update counts
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]
            acc = n_correct / n_total
            test_acc.append(acc)

            loss = loss_func(output, targets)
            test_loss.append(loss.item())

        test_acc = np.mean(test_acc)
        test_accs[it] = test_acc
        test_loss = np.mean(test_loss)
        test_losses[it] = test_loss

        dt = datetime.now() - t_0
        print(f' Epoch {it+1}/{epochs}, Train Loss:{train_loss: .4f}, Test Loss:{test_loss: .4f} , Duration: {dt}')

    
    torch.save(model.state_dict(), save_path)

    return train_losses, test_losses, train_accs, test_accs