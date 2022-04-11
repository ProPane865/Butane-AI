import discord
from discord.ext import commands
import requests

from matplotlib import image
import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import img_rec.neural_network
import img_rec.dataset_instantiate
import img_rec.dataset_multi_instantiate
import img_rec.testing
import img_rec.image_scraper

import os
import pickle
import glob

if __name__ == "__main__":

    if not os.path.exists("img_rec/data"):
        os.makedirs("img_rec/data")

        if not os.path.exists("img_rec/data/bot"):
            os.makedirs("img_rec/data/bot")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    intents = discord.Intents.default()
    intents.members = True

    prefix = "-"
    bot = commands.Bot(command_prefix=prefix, intents=intents)

    classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

    net = img_rec.neural_network.SpinalResNet18().to(device)

    try:
        net.load_state_dict(torch.load("img_rec/traindata/model_weights96.pth"))
        net.eval()
    except:
        pass

    itemlist = []

    try:
        with open("img_rec/datasets", "rb") as fp:
            itemlist = pickle.load(fp)
    except:
        itemlist = []

    @bot.event
    async def on_ready():
        print("Logged in as {0.user}".format(bot))

    @bot.command(name="testcommand")
    async def testcommand(ctx):
        await ctx.send("testcommand response")

    @bot.command(name="traindata")
    async def traindata(ctx):
        if ctx.message.author.id in [343239616435847170, 864572010574118932]:

            try:
                with open("img_rec/datasets", "rb") as fp:
                    global itemlist
                    itemlist = pickle.load(fp)
                    ds = torch.utils.data.ConcatDataset(itemlist)
                    ds_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=2)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

                for epoch in range(14):
                    img_rec.testing.train_set_disc(ds_loader, net, criterion, optimizer, device, epoch)

                itemlist = []
                with open("img_rec/datasets", "wb") as fp:
                    pickle.dump(itemlist, fp)

                dir = "img_rec/data/bot"
                filelist = glob.glob(os.path.join(dir, "*"))

                for f in filelist:
                    os.remove(f)

                try:
                    torch.save(net.state_dict(), "img_rec/traindata/model_weights96.pth")
                except:
                    if not os.path.exists("img_rec/traindata"):
                        os.makedirs("img_rec/traindata")
                    open("img_rec/traindata/model_weights96.pth", "w+").close()
                    torch.save(net.state_dict(), "img_rec/traindata/model_weights96.pth")

                await ctx.send("Finished training using cached data")

            except:
                await ctx.send("The cache is currently empty")

        else:
            await ctx.send("You do not have permission to use that command")

    @bot.command(name="clearcache")
    async def clearcache(ctx):
        if ctx.message.author.id in [343239616435847170, 864572010574118932]:

            global itemlist
            itemlist = []
            with open("img_rec/datasets", "wb") as fp:
                pickle.dump(itemlist, fp)
            
            dir = "img_rec/data/bot"
            filelist = glob.glob(os.path.join(dir, "*"))

            for f in filelist:
                os.remove(f)

            await ctx.send("Finished clearing cache data")

        else:
            await ctx.send("You do not have permission to use that command")

    @bot.command(name="cache")
    async def cache(ctx):
        if ctx.message.author.id in [343239616435847170, 864572010574118932]:

            try:
                with open("img_rec/datasets", "rb") as fp:
                    global itemlist
                    itemlist = pickle.load(fp)
                    item_len = len(itemlist)

                await ctx.send(f"There are currently {item_len} items in the cache")

            except:
                await ctx.send("The cache is currently empty")

        else:
            await ctx.send("You do not have permission to use that command")

    @bot.command(name="undocache")
    async def undocache(ctx):
        if ctx.message.author.id in [343239616435847170, 864572010574118932]:

            try:
                with open("img_rec/datasets", "rb") as fp:
                    global itemlist
                    itemlist = pickle.load(fp)

                for i in range(3):
                    itemlist.pop()

                with open("img_rec/datasets", "wb") as fp:
                    pickle.dump(itemlist, fp)
                    
                await ctx.send("Removed last item from cache")

            except:
                await ctx.send("The cache is currently empty")

        else:
            await ctx.send("You do not have permission to use that command")

    @bot.command(name="evaluate")
    async def evaluate(ctx, label):
        try:
            print(str(ctx.message.attachments[0]))
            if str(ctx.message.attachments[0])[0:26] == "https://cdn.discordapp.com":
                r = requests.get(str(ctx.message.attachments[0]), stream=True)

                rnum = str(random.randrange(0, 3000))
                
                with open(f"img_rec/data/bot/image{rnum}.jpg", "wb") as out_file:
                    out_file.write(r.content)
                    await ctx.send("New image created successfully!")

                res_dataset0 = img_rec.dataset_instantiate.InstantiateDatasetAug0(f"img_rec/data/bot/image{rnum}.jpg", label)
                res_dataset1 = img_rec.dataset_instantiate.InstantiateDatasetAug1(f"img_rec/data/bot/image{rnum}.jpg", label)
                res_dataset2 = img_rec.dataset_instantiate.InstantiateDatasetAug2(f"img_rec/data/bot/image{rnum}.jpg", label)
                res_dataset = torch.utils.data.ConcatDataset([res_dataset0, res_dataset1, res_dataset2])
                res_data = torch.utils.data.DataLoader(res_dataset, batch_size=1, shuffle=True, num_workers=2)
                res = img_rec.testing.test_set_disc(res_data, net, device, classes)

                global itemlist
                itemlist.append(res_dataset0)
                itemlist.append(res_dataset1)
                itemlist.append(res_dataset2)

                with open("img_rec/datasets", "wb") as fp:
                    pickle.dump(itemlist, fp)

                await ctx.send(f"Prediction: {res}")

        except IndexError:
            await ctx.send("Please enter a valid image.")

    @bot.command(name="scrape")
    async def scrape(ctx, data, traindata, size):
        if ctx.message.author.id in [343239616435847170, 864572010574118932]:
            await ctx.send(f"Scraping {size} '{data}' images...")

            scraper = img_rec.image_scraper.ImageScraper("img_rec/data/scrape")
            images = scraper.scrape(data, traindata, int(size))

            scraped_dataset0 = img_rec.dataset_multi_instantiate.InstantiateMultiDatasetAug0("img_rec/data/scrape", traindata)
            scraped_dataset1 = img_rec.dataset_multi_instantiate.InstantiateMultiDatasetAug1("img_rec/data/scrape", traindata)
            scraped_dataset2 = img_rec.dataset_multi_instantiate.InstantiateMultiDatasetAug2("img_rec/data/scrape", traindata)

            global itemlist
            itemlist.append(scraped_dataset0)
            itemlist.append(scraped_dataset1)
            itemlist.append(scraped_dataset2)

            with open("img_rec/datasets", "wb") as fp:
                pickle.dump(itemlist, fp)

            await ctx.send(f"Finished scraping {images} {data} images for the {traindata} class!")

        else:
            await ctx.send("You do not have permission to use that command")

    bot.run(open("token", "r").read())