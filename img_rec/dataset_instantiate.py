import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def square(img, bg_color):
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), bg_color)
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(img.mode, (height, height), bg_color)
        result.paste(img, ((height - width) // 2, 0))
        return result

class InstantiateDatasetAug0(Dataset):
	def __init__(self, img_path, class_name):
		self.imgs_path = str(img_path)
		self.data = []
		self.data.append([img_path, class_name])
		print(self.data)
		self.class_map = {"airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9}
		self.img_dim = (96, 96)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, class_name = self.data[idx]
		img = Image.open(img_path).convert("RGB")
		img = square(img, "black")
		img = img.resize(self.img_dim)
		class_id = self.class_map[class_name]
		img_tensor = transforms.ToTensor()(img)
		class_id = torch.tensor([class_id])
		return img_tensor, class_id

class InstantiateDatasetAug1(Dataset):
	def __init__(self, img_path, class_name):
		self.imgs_path = str(img_path)
		self.data = []
		self.data.append([img_path, class_name])
		print(self.data)
		self.class_map = {"airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9}
		self.img_dim = (96, 96)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, class_name = self.data[idx]
		img = Image.open(img_path).convert("RGB")
		img = square(img, "black")
		img = img.resize(self.img_dim)
		class_id = self.class_map[class_name]
		img = transforms.RandomVerticalFlip()(img)
		img = transforms.RandomHorizontalFlip()(img)
		img_tensor = transforms.ToTensor()(img)
		class_id = torch.tensor([class_id])
		return img_tensor, class_id

class InstantiateDatasetAug2(Dataset):
	def __init__(self, img_path, class_name):
		self.imgs_path = str(img_path)
		self.data = []
		self.data.append([img_path, class_name])
		print(self.data)
		self.class_map = {"airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9}
		self.img_dim = (96, 96)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, class_name = self.data[idx]
		img = Image.open(img_path).convert("RGB")
		img = square(img, "black")
		img = img.resize(self.img_dim)
		class_id = self.class_map[class_name]
		img = transforms.RandomRotation(45, expand=False)(img)
		img_tensor = transforms.ToTensor()(img)
		class_id = torch.tensor([class_id])
		return img_tensor, class_id