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
import random

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
    epochs = 160

    torch.manual_seed(0)
    random.seed(0)

    trainset_a = torchvision.datasets.STL10(root='./predata_sets/96', download=True, transform=transform_a)
    trainset_b = torchvision.datasets.STL10(root='./predata_sets/96', download=True, transform=transform_b)
    trainset_c = torchvision.datasets.STL10(root='./predata_sets/96', download=True, transform=transform_c)
    trainset = torch.utils.data.ConcatDataset([trainset_a, trainset_b, trainset_c])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    def update_lr(optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    total_step = len(trainloader)

    classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

    net = neural_network.SpinalResNet18().to(device)

    try:
        net.load_state_dict(torch.load("./traindata/model_weights96.pth"))
        net.eval()
    except:
        pass

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        testing.train_set(trainloader, net, criterion, optimizer, device, epoch)
    print("Finished Training")

    try:
        torch.save(net.state_dict(), "./traindata/model_weights96.pth")
    except:
        if not os.path.exists("./traindata"):
            os.makedirs("./traindata")
        open("./traindata/model_weights96.pth", "w+").close()
        torch.save(net.state_dict(), "./traindata/model_weights96.pth")