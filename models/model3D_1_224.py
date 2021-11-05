import torch
import torch.nn as nn


class VGGStyle3DCNN(nn.Module):
    """
    - A VGG-style 3D CNN with 11 layers.
    - Kernel size is kept 3 for all three dimensions - (time, H, W)
      except the first layer has kernel size of (3, 5, 5)
    - Time dimension is preserved with `padding=1` and `stride=1` (except a
      stride of 2 after third conv layer), and is averaged at the end

    Arguments:
    - Input: a (batch_size, 3, sequence_length, W, H) tensor
    - Returns: a (batch_size, 512) sized tensor
    """

    def __init__(self):
        super(VGGStyle3DCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.Dropout3d(p=0.2),
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block4 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # get convolution column features

        x = self.block1(x)
        # print(x.size())
        x = self.block2(x)
        # print(x.size())
        x = self.block3(x)
        # print(x.size())
        x = self.block4(x)
        # print(x.size())
        x = self.block5(x)
        # print(x.size())

        # averaging features in time dimension
        # x = x.mean(-1).mean(-1).mean(-1)

        return x


if __name__ == "__main__":
    num_classes = 157
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 32, 224, 224))
    model = VGGStyle3DCNN()
    output = model(input_tensor)
    print(output.size())
