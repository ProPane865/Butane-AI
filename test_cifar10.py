import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import neural_network

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

    trainset_a = torchvision.datasets.CIFAR10(root='./predata_sets', train=True, download=True, transform=transform_a)
    trainset_b = torchvision.datasets.CIFAR10(root='./predata_sets', train=True, download=True, transform=transform_b)
    trainset_c = torchvision.datasets.CIFAR10(root='./predata_sets', train=True, download=True, transform=transform_c)
    trainset = torch.utils.data.ConcatDataset([trainset_a, trainset_b, trainset_c])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./predata_sets', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = neural_network.NeuralNetwork().to(device)

    try:
        net.load_state_dict(torch.load("./traindata/model_weights.pth"))
        net.eval()
    except:
        pass

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def test_loop(dataloader, model):
        dataiter = iter(dataloader)
        images, labels = dataiter.next()

        imshow(torchvision.utils.make_grid(images))
        print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            for data in dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)

                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        overall_accuracy = 0

        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
            overall_accuracy += accuracy / len(correct_pred.items())
        print(f"Overall accuracy is: {overall_accuracy:.1f} %")

    test_loop(testloader, net)
    test_loop(trainloader, net)
    print("Finished Testing")