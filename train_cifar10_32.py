import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import neural_network
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import testing

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
    epochs = 7

    trainset_a = torchvision.datasets.CIFAR10(root='./predata_sets/32', train=True, download=True, transform=transform_a)
    trainset_b = torchvision.datasets.CIFAR10(root='./predata_sets/32', train=True, download=True, transform=transform_b)
    trainset_c = torchvision.datasets.CIFAR10(root='./predata_sets/32', train=True, download=True, transform=transform_c)
    trainset = torch.utils.data.ConcatDataset([trainset_a, trainset_b, trainset_c])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = neural_network.NeuralNetwork().to(device)

    try:
        net.load_state_dict(torch.load("./traindata/model_weights.pth"))
        net.eval()
    except:
        pass

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):
        testing.train_set(trainloader, net, criterion, optimizer, device, epoch)
    print("Finished Training")

    try:
        torch.save(net.state_dict(), "./traindata/model_weights.pth")
    except:
        if not os.path.exists("./traindata"):
            os.makedirs("./traindata")
        open("./traindata/model_weights.pth", "w+").close()
        torch.save(net.state_dict(), "./traindata/model_weights.pth")