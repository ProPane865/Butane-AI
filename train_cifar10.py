import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import neural_network
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform_a = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_b = transforms.Compose([
        transforms.RandomRotation(50, expand=False),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_c = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4
    learning_rate = 1e-3
    epochs = 10

    trainset_a = torchvision.datasets.CIFAR10(root='./predata_sets', train=True, download=True, transform=transform_a)
    trainset_b = torchvision.datasets.CIFAR10(root='./predata_sets', train=True, download=True, transform=transform_b)
    trainset_c = torchvision.datasets.CIFAR10(root='./predata_sets', train=True, download=True, transform=transform_c)
    trainset = torch.utils.data.ConcatDataset([trainset_a, trainset_b, trainset_c])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = neural_network.NeuralNetwork().to(device)

    try:
        net.load_state_dict(torch.load("./traindata/model_weights.pth"))
        net.eval()
    except:
        pass

    def train_loop(dataloader, model, loss_fn, optimizer):
        model.train()
        running_loss = 0.0
        for batch, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch % 2000 == 1999:
                print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):
        train_loop(trainloader, net, criterion, optimizer)
    print("Finished Training")

    try:
        torch.save(net.state_dict(), "./traindata/model_weights.pth")
    except:
        if not os.path.exists("./traindata"):
            os.makedirs("./traindata")
        open("./traindata/model_weights.pth", "w+").close()
        torch.save(net.state_dict(), "./traindata/model_weights.pth")