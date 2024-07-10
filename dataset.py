import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch


class DresdenDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        basename = os.path.basename(image_dir)
        self.label_dir = os.path.join((os.path.dirname(image_dir)),
                                      basename, 'merged')
        self.transform = transform
        self.image_names = [filename for filename in os.listdir(image_dir) if filename.startswith("image")]
        self.label_names = [filename for filename in os.listdir(image_dir) if filename.startswith("mask")]

        # Sort the image and label names based on the numeric part extracted from filenames
        self.image_names.sort(key=lambda x: int(x[5:7]))  # Extract the two-digit number from "imageXX.png"
        self.label_names.sort(key=lambda x: int(x[4:6]))  # Extract the two-digit number from "maskXX.png"

        # print("image_names:", self.image_names)
        # print("label:", self.label_names)

    def __len__(self):
        return len(self.image_names)

    def classes(self):
        return torch.Tensor([0,1])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = np.array(Image.open(os.path.join(self.image_dir, self.image_names[idx])).convert('RGB'))
        label1 = np.array(Image.open(os.path.join(self.image_dir, self.label_names[idx])).convert('RGB'))
        # to use just three conditions, we create another label with np.zeros
        label = np.zeros(np.shape(label1), np.uint8)[:,:,0:2]
        label[:,:,0][label1[:,:,0]>125] = 1
        label[:,:,1][label1[:,:,0]<125] = 1

        dictionary = {'image0': image, 'image1': label}

        if self.transform is not None:
            dictionary = self.transform(dictionary)

        return dictionary
