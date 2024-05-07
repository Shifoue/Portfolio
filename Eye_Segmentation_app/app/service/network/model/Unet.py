import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.depht = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, X):
        return self.depht(X)

class myUNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
      super(myUNET, self).__init__()
      self.encode = nn.ModuleList()
      self.decode = nn.ModuleList()
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #divide size per 2

      # Encode
      for feature in features:
          self.encode.append(DoubleConv(in_channels, feature))
          in_channels = feature

      # Decode
      for feature in reversed(features):
          self.decode.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
          self.decode.append(DoubleConv(feature*2, feature))

      self.bottleneck = DoubleConv(features[-1], features[-1]*2)
      self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, X):
        skip_connections = []

        for depht in self.encode:
            X = depht(X)
            skip_connections.append(X)
            X = self.pool(X)

        X = self.bottleneck(X)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.decode), 2):
            X = self.decode[i](X)
            #skip_connection = skip_connections[-1 - i//2]
            skip_connection = skip_connections[i//2]

            if X.shape != skip_connection.shape:
                X = torch.nn.functional.interpolate(X, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, X), dim=1)
            X = self.decode[i+1](concat_skip)

        return self.final_conv(X)