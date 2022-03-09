import discord
import requests
import random
import numpy as np
import torch
import neural_network
import dataset_instantiate
import testing
import os
import torch.nn as nn
import torch.optim as optim
import pickle

if __name__ == "__main__":

    if not os.path.exists("./data"):
        os.makedirs("./data")

        if not os.path.exists("./data/bot"):
            os.makedirs("./data/bot")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    client = discord.Client()
    prefix = "-"

    classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

    net = neural_network.NeuralNetwork96().to(device)

    try:
        net.load_state_dict(torch.load("./traindata/model_weights96.pth"))
        net.eval()
    except:
        pass

    @client.event
    async def on_ready():
        print("Logged in as {0.user}".format(client))

    @client.event
    async def on_message(message):
        itemlist = []

        try:
            with open("datasets", "rb") as fp:
                itemlist = pickle.load(fp)
        except:
            itemlist = []

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

                    res_dataset = dataset_instantiate.InstantiateDataset(f"data/bot/image{rnum}.jpg", message.content[10:])
                    res_data = torch.utils.data.DataLoader(res_dataset, batch_size=1, shuffle=True, num_workers=2)
                    res = testing.test_set_disc(res_data, net, device, classes)

                    itemlist.append(res_dataset)

                    with open("datasets", "wb") as fp:
                        pickle.dump(itemlist, fp)

                    await message.channel.send(f"Prediction: {res}")

            except IndexError:
                await message.channel.send("Please enter a valid image.")

        if message.content.startswith(prefix+"traindata"):

            with open("datasets", "rb") as fp:
                itemlist = pickle.load(fp)
                ds = torch.utils.data.ConcatDataset(itemlist)
                ds_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=2)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

            for epoch in range(14):
                testing.train_set(ds_loader, net, criterion, optimizer, device, epoch)

            await message.client.send("Finished training using cached data")

    client.run(open("token", "r").read())