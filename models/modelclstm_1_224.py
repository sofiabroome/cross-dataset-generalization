import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self._batch_size, self._state_height, self._state_width = b_h_w
        # if using requires_grad flag,
        # torch.save will not save parameters indeed although it may be updated every epoch.
        # However, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        # These three tensors should be the same dimension as the cell state.
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))# .to(device)
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))# .to(device)
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))# .to(device)
        self._input_channel = input_channel
        self._num_filter = num_filter

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    def forward(self, seq_len, inputs=None, states=None):

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float)  #.to(device)
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float)  #.to(device)
        else:
            h, c = states

        outputs = []
        for time_step in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float) #.to(device)
            else:
                x = inputs[time_step, :, :, :, :]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
            # Torch stack: Concatenates a sequence of tensors along a new dimension.
        return torch.stack(outputs), (h, c)


class TimeDistMaxPooling2D(nn.Module):
    def __init__(self):
        super(TimeDistMaxPooling2D, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=2)

    def forward(self, seq_len, inputs):
        outputs = []
        for time_step in range(seq_len):
            outputs.append(self.mp(inputs[time_step, :]))
        return torch.stack(outputs)


class ConvLSTMBlock(nn.Module):
    def __init__(self, b_h_w, batch_size):
        super(ConvLSTMBlock, self).__init__()
        self.convlstm = ConvLSTM(input_channel=3, num_filter=3, b_h_w=b_h_w, kernel_size=3)
        self.timedist_mp = TimeDistMaxPooling2D()
        self.bn3d = nn.BatchNorm3d(num_features=batch_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):
        x, (_, _) = self.convlstm(seq_len=x.shape[0], inputs=x)
        print(x.size())
        x = self.timedist_mp(seq_len=x.shape[0], inputs=x)
        print(x.size())
        x = self.bn3d(x)
        print(x.size())
        return x


class ConvLSTMModel(nn.Module):
    """

    Arguments:
    - Input: a (batch_size, 3, sequence_length, W, H) tensor
    - Returns: a (batch_size, 512) sized tensor
    """

    def __init__(self, config):
        super(ConvLSTMModel, self).__init__()
        self.block1 = ConvLSTMBlock(b_h_w=(1, 224, 224), batch_size=config['clip_size'])
        self.block2 = ConvLSTMBlock(b_h_w=(1, 112, 112), batch_size=config['clip_size'])
        self.block3 = ConvLSTMBlock(b_h_w=(1, 56, 56), batch_size=config['clip_size'])
        self.linear = nn.Linear(14112, out_features=config['num_classes'])
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        # get convolution column features

        x = x[0]
        x = self.block1(x)
        print(x.size())
        x = self.block2(x)
        print(x.size())
        x = self.block3(x)
        print(x.size())

        # averaging features in time dimension
        # x = x.mean(-1).mean(-1).mean(-1)
        x = self.flatten(x)
        x = self.linear(x)
        print(x.size())

        return x


if __name__ == "__main__":
    num_classes = 48
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.autograd.Variable(torch.rand(72, 1, 3, 224, 224)).to(device)
    model = ConvLSTMModel()
    output = model(x=input_tensor)
    # print(output[0].size())
    # print(output[1][0].size())
    # print(output[1][1].size())
