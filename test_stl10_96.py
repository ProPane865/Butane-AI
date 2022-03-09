import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import neural_network
import testing

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    batch_size = 4

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

    trainset_a = torchvision.datasets.STL10(root='./predata_sets/96', download=True, transform=transform_a)
    trainset_b = torchvision.datasets.STL10(root='./predata_sets/96', download=True, transform=transform_b)
    trainset_c = torchvision.datasets.STL10(root='./predata_sets/96', download=True, transform=transform_c)
    trainset = torch.utils.data.ConcatDataset([trainset_a, trainset_b, trainset_c])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = torchvision.datasets.STL10(root='./predata_sets/96', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

    net = neural_network.NeuralNetwork96().to(device)

    try:
        net.load_state_dict(torch.load("./traindata/model_weights96.pth"))
        net.eval()
    except:
        pass

    testing.test_set(testloader, net, device, classes, batch_size)
    testing.test_set(trainloader, net, device, classes, batch_size)
    print("Finished Testing")