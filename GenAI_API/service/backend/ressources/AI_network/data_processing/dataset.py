import random
from PIL import Image

seed = 42

class CatDogDataset():
    def __init__(self, data, transforms, target_transform=None):
        self.data =  data #(image_name, label_name)
        self.transforms = transforms
        self.target_transform = target_transform

        # child_directories = next(os.walk(dataset_path))[1] #These will be the labels
        # for label in child_directories:
        #     path = os.join(dataset_path, label)

        #     self.data.append((os.join(path, f), label) for f in os.listdir(path) if os.isfile(os.join(path, f)))

        random.shuffle(self.data) #Doing so avoid the model learning 4000 firsts are class 0, 4000 seconds are class 1, etc...

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label