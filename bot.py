import discord
import requests
import random
import numpy as np
import torch
import neural_network
import dataset_instantiate

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    client = discord.Client()
    prefix = "-"

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = neural_network.NeuralNetwork().to(device)

    try:
        net.load_state_dict(torch.load("./traindata/model_weights.pth"))
        net.eval()
    except:
        pass

    def test_loop(dataloader, model):
        net.eval()

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

    @client.event
    async def on_ready():
        print("Logged in as {0.user}".format(client))

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        if message.content.startswith(prefix+"testcommand"):
            await message.channel.send("testcommand response")

        if message.content.startswith(prefix+"evaluate"):

            try:
                print(str(message.attachments[0]))
                if str(message.attachments[0])[0:26] == "https://cdn.discordapp.com":
                    r = requests.get(str(message.attachments[0]), stream=True)

                    rnum = str(random.randrange(0, 3000))
                    
                    with open(f"data/bot/image{rnum}.jpg", "wb") as out_file:
                        out_file.write(r.content)
                        await message.channel.send("New image created successfully!")

                    res = test_loop(torch.utils.data.DataLoader(dataset_instantiate.InstantiateDataset(f"data/bot/image{rnum}.jpg", "car"), batch_size=1, shuffle=True, num_workers=2), net)

                    await message.channel.send(f"Prediction: {res}")
                        

            except IndexError:
                await message.channel.send("Please enter a valid image.")

    client.run(open("token", "r").read())