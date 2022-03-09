import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch

def train_set(dataloader, model, loss_fn, optimizer, device, epoch):
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

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test_set(dataloader, model, device, classes, batch_size):
    model.eval()

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

def test_set_disc(dataloader, model, device, classes):
    model.eval()

    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    x = None

    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(1)))

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                x = classes[prediction]

    return x

def train_set_disc(dataloader, model, loss_fn, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for batch, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, torch.max(labels, 1)[0])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss