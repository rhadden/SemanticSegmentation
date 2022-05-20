import torch.nn as nn
import torch


class Unet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(64),
                self.relu,
            ),

            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(128),
                self.relu,

            ),

            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(256),
                self.relu,
            ),

            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(512),
                self.relu,
            ),
        ])

        self.upConvs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(256),
                self.relu,
            ),

            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(128),
                self.relu,
            ),

            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(64),
                self.relu,
            ),
        ])
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(256),
                self.relu,
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(128),
                self.relu,
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(64),
                self.relu,
            ),
        ])

        # Final output layer to have a 30 output classes
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

    def forward(self, x):

        skipConnections = []
        for layer in self.encoder[:-1]:
            # Can use this to append to layers on reverse operations
            x = layer(x)
            skipConnections.append(x)
            x = nn.MaxPool2d(2, 2)(x)

        # Final encoding without maxpool/skip connection
        x = self.encoder[-1](x)

        for i, layer in enumerate(self.decoder):
            # First upscale
            x = self.upConvs[i](x)

            # Merge with corresponding output
            x = torch.cat([skipConnections.pop(), x], axis=1)

            # Convolutions
            x = layer(x)

        del skipConnections
        del layer
        pred = self.classifier(x)
        del x
        return pred
