# Template created by Sam Lerman, slerman@ur.rochester.edu.

from torchvision.datasets import mnist  # Dataset
from torch.utils.data import DataLoader  # Gets batches, parallel workers can speed up hard disk data reads
from torchvision.transforms import ToTensor, Normalize, Compose  # Can pre-process images

import torch  # Pytorch!
from torch import nn  # Neural networks
from torch.optim import SGD  # Stochastic gradient descent -- optimize the neural networks


epochs = 2  # How many times to iterate through the full data for training
batch_size = 32  # How many data-points to feed into the neural network at a time
lr = 1e-2  # Learning rate -- controls the magnitude of gradients

# Pre-process
data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

# Get data
train_dataset = mnist.MNIST(root='./', train=True, transform=data_transform, download=True)
test_dataset = mnist.MNIST(root='./', train=False, transform=data_transform, download=True)

# Divide data into batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)

# The neural network
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                      nn.Linear(128, 64), nn.ReLU(),
                      nn.Linear(64, 10))

model.to('cpu')  # Can write 'cuda' for GPU if you have one!

# The loss function and optimizer
cost = nn.CrossEntropyLoss()
optim = SGD(model.parameters(), lr=lr)

correct = total = 0

# Start learning and evaluating
for epoch in range(epochs):

    # Just sets model.training to True. Some neural networks behave differently during training (e.g. nn.Dropout).
    # Here, makes no difference. Just convention
    model.train()

    # Train on the training data
    for i, (x, y) in enumerate(train_loader):
        # Our neural network accepts flat 1D inputs, so flatten the 2D RGB/or grayscale images
        x = torch.flatten(x, start_dim=1).float()  # Many Pytorch modules expect float data types by default

        y_pred = model(x)  # Predict a class
        loss = cost(y_pred, y)  # Compute error

        # Tally scores
        correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
        total += y.shape[0]

        # Print scores
        if i % 1000 == 0:
            print('Epoch: {}, Acc: {}/{} ({:.0f}%)'.format(epoch, correct, total, 100. * correct / total))

            correct = total = 0

        # Optimize the neural network - learn!
        optim.zero_grad()  # Resets model's internal gradients to zero
        loss.backward()  # Adds the new gradients into memory (by computing them via the backpropagation function)
        optim.step()  # Steps those gradients on the model

    correct = total = 0  # Reset score statistics

    model.eval()  # Sets model.training to False

    # Evaluate scores on the evaluation data
    for i, (x, y) in enumerate(test_loader):
        x = torch.flatten(x, start_dim=1).float()
        y_pred = model(x).detach()

        correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
        total += y.shape[0]

    print('Epoch: {}, Accuracy: {}/{} ({:.0f}%)'.format(epoch, correct, total, 100. * correct / total))
