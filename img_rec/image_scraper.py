import os
import requests
from bs4 import BeautifulSoup

class ImageScraper():
    def __init__(self, destination):
        self.source = "https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&"
        self.user_agent = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}
        self.destination = destination

    def scrape(self, data, traindata, size):
        if not os.path.exists(self.destination):
            os.mkdir(self.destination)
        
        url = self.source + 'q=' + data
        response = requests.get(url, headers=self.user_agent)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        results = soup.findAll('img', {'class': 'rg_i Q4LuWd'})

        count = 1
        links = []

        for result in results:
            try:
                link = result['data-src']
                links.append(link)
                count += 1
                if(count > size):
                    break

            except KeyError:
                continue

        print(f"Downloading {len(links)} images...")

        for i, link in enumerate(links):
            response = requests.get(link)

            image_name = self.destination + '/' + traindata + str(i+1) + '.jpg'

            with open(image_name, 'wb') as fh:
                fh.write(response.content)

        return len(links)