import torch
import torch.nn as nn
from torchvision import transforms

class ImageScaling(nn.Module):
    def __init__(self):
        super(ImageScaling, self).__init__()

    def forward(self, x):
        for i in range(len(x)):
            x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())

        return x


def conversion_block():
    block = nn.Sequential(
        nn.ConvTranspose2d(1, 128, kernel_size=5),
        nn.BatchNorm2d(128),
        ImageScaling(),
        nn.ConvTranspose2d(128, 64, kernel_size=5),
        nn.BatchNorm2d(64),
        ImageScaling(),
        nn.ConvTranspose2d(64, 32, kernel_size=5),
        nn.BatchNorm2d(32),
        ImageScaling(),
    )

    return block

def generator_block(input_dim, output_dim, kernel_size):
    block = nn.Sequential(
        nn.ConvTranspose2d(input_dim, output_dim, kernel_size),
        nn.BatchNorm2d(output_dim),
        #nn.ReLU(inplace=True)
        ImageScaling(),

    )

    return block

class Generator(nn.Module):
    def __init__(self, vocabulary):
        super(Generator, self).__init__()
        self.voc = vocabulary
        self.embs = nn.Embedding(len(vocabulary.keys()), len(vocabulary.keys())) # 2x2 matrix
        self.first_transpose = conversion_block() #create a start image from the word : image [1, 14, 14]
        self.layer1 = generator_block(32, 16, kernel_size=19) # image [1, 32, 32]
        self.layer2 = generator_block(16, 8, kernel_size=33) # image [1, 64, 64]
        self.layer3 = generator_block(8, 4, kernel_size=65) # image [1, 128, 128]
        self.layer4 = generator_block(4, 2, kernel_size=129) # image [1, 256, 256]
        self.layer5 = generator_block(2, 1, kernel_size=257) # image [1, 512, 512]
        self.layer_image_normalization = transforms.ToTensor()

    def forward(self, word):
        word_encoded = [0 for _ in range(len(self.voc.keys()))]
        word_encoded[self.voc[word]] = 1
        word_encoded = torch.tensor([[word_encoded]])

        x = self.embs(word_encoded)
        x = self.first_transpose(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return x[0]