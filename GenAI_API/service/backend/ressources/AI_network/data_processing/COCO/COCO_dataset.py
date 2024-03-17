import random
from PIL import Image

seed = 42

class COCODataset():
    def __init__(self, data, transforms, target_transform=None):
        self.data =  data #(image_path, annotation sentence)
        self.transforms = transforms
        self.target_transform = target_transform

        random.shuffle(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, annotation = self.data[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            annotation = self.target_transform(annotation)
        return image, annotation