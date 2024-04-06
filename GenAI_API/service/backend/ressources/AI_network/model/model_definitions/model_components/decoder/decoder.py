import torch
import torch.nn as nn

class BlockConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockConv2D, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels

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

class BlockConv2D_Upsample2D(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, up_in_channels, up_out_channels):
        super(BlockConv2D_Upsample2D, self).__init__()
        self.in_features = conv_in_channels
        self.out_features = up_out_channels

        self.conv = BlockConv2D(conv_in_channels, conv_out_channels)
        self.upsample = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)

        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, input_resize_dim, in_features, out_features=1):
        super(Decoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, input_resize_dim) #nn.Conv1d(input_dim, input_resize_dim, 1)

        self.block1 = BlockConv2D_Upsample2D(in_features, in_features, in_features, in_features)
        self.block2 = BlockConv2D_Upsample2D(self.block1.out_features, self.block1.out_features, self.block1.out_features, self.block1.out_features // 2)
        self.block3 = BlockConv2D_Upsample2D(self.block2.out_features, self.block2.out_features, self.block2.out_features, self.block2.out_features // 2)
        self.block4 = BlockConv2D_Upsample2D(self.block3.out_features, self.block3.out_features, self.block3.out_features, self.block3.out_features // 2)
        self.block5 = BlockConv2D_Upsample2D(self.block4.out_features, self.block4.out_features, self.block4.out_features, self.block4.out_features // 2)

        self.output_layer = nn.Conv2d(self.block5.out_features, out_features, kernel_size=1)

    def forward(self, x):
        #input shape : batch size, 768, 512
        x = self.input_linear(x)
        #shape : batch size, 1024, 512
        x = torch.reshape(x, shape=(x.shape[0], -1, 32, 32))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.output_layer(x)

        return x