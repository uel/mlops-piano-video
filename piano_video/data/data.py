from pathlib import Path
from functools import partial

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T

from PIL import Image


def exists(val):
    return val is not None

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        convert_image_to_type = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        convert_fn = partial(convert_image_to, convert_image_to_type) if exists(convert_image_to_type) else nn.Identity()

        landmark_path = Path(folder).parent / 'landmarks.pt'
        self.landmarks = torch.load(landmark_path).float()

        self.transform = T.Compose([
            T.Lambda(convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img), self.landmarks[index]