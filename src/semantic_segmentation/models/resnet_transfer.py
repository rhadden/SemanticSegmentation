import torch
from src.semantic_segmentation.utils.init_weights import init_weights
class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        news = self.shape.copy()
        news[0] = x.shape[0]

        return x.view(*tuple(news))


def getTransferModel(n_class, batch_size):
    decoder = torch.nn.Sequential(
        Reshape([batch_size, 512, 16, 32]),

        torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        ),

        torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        ),

        torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        ),

        torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        ),

        torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ),

        torch.nn.Conv2d(32, n_class, kernel_size=1)
    )
    decoder.apply(init_weights)

    model = models.resnet34(pretrained=True)

    for param in model.parameters():
        # False implies no retraining
        param.requires_grad = False

    del param

    model.avgpool = torch.nn.Identity()
    model.fc = decoder
    return model