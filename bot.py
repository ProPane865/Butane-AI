import discord
import requests
import random
import numpy as np
import torch
import neural_network
import dataset_instantiate
import testing

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

                    res_dataset = dataset_instantiate.InstantiateDataset(f"data/bot/image{rnum}.jpg", "car")
                    res_data = torch.utils.data.DataLoader(res_dataset, batch_size=1, shuffle=True, num_workers=2)
                    res = testing.test_set_disc(res_data, net, device, classes)

                    await message.channel.send(f"Prediction: {res}")

            except IndexError:
                await message.channel.send("Please enter a valid image.")

    client.run(open("token", "r").read())