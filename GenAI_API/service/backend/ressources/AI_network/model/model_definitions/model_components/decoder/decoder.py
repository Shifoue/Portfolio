import torch
import torch.nn as nn

class BlockConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockConv2D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
      return self.double_conv(x)

# class BlockUpsamble2D(nn.Module):

class BlockConv2D_Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockConv2D_Upsample2D, self).__init__()
        self.conv = BlockConv2D(in_channels, in_channels)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)

        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, input_resize_dim):
        super(Decoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, input_resize_dim) #nn.Conv1d(input_dim, input_resize_dim, 1)
        self.block1 = BlockConv2D_Upsample2D(512, 256)

        self.block2 = BlockConv2D_Upsample2D(256, 64)

        #self.block3 = BlockConv2D_Upsample2D(128, 64)

        self.block4 = BlockConv2D_Upsample2D(64, 16)

        #self.block5 = BlockConv2D_Upsample2D(32, 16)

        self.block6 = BlockConv2D_Upsample2D(16, 4)

        #self.block7 = BlockConv2D_Upsample2D(8, 4)

        #self.block8 = BlockConv2D_Upsample2D(4, 2)

        self.block9 = BlockConv2D_Upsample2D(4, 1)

    def forward(self, x):

        #input shape : batch size, 768, 512
        x = self.input_linear(x)
        #shape : batch size, 1024, 512
        x = torch.reshape(x, shape=(x.shape[0], -1, 32, 32))

        x = self.block1(x)
        x = self.block2(x)
        #x = self.block3(x)
        x = self.block4(x)
        #x = self.block5(x)
        x = self.block6(x)
        #x = self.block7(x)
        #x = self.block8(x)
        x = self.block9(x)

        return x