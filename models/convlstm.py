import torch
from torch import nn
import pytorch_lightning as pl

# The implementation is adapted from
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py


class ConvLSTMModel(pl.LightningModule):
    def __init__(self, input_channels, hidden_per_layer, kernel_size_per_layer,
                 return_all_layers=False, batch_first=True):
        super(ConvLSTMModel, self).__init__()

        self.hidden_per_layer = hidden_per_layer
        self.input_channels = input_channels
        self.num_layers = len(hidden_per_layer)
        self.return_all_layers = return_all_layers
        self.batch_first = batch_first
        self.blocks = []

        assert(len(hidden_per_layer) == len(kernel_size_per_layer))

        for i, nb_channels in enumerate(self.hidden_per_layer):
            cur_input_dim = self.input_channels if i == 0 else self.hidden_per_layer[i - 1]
            self.blocks.append(ConvLSTMBlock(cur_input_dim, hidden_dim=hidden_per_layer[i],
                                             kernel_size=kernel_size_per_layer[i], bias=True))
        self.conv_lstm_blocks = nn.ModuleList(self.blocks)

    def forward(self, input_tensor, hidden_state=None):
        """
         Parameters
         ----------
         input_tensor: todo
             5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
         hidden_state: todo
             None. todo implement stateful
         Returns
         -------
         last_state_list, layer_output
         """

        # find size of different input dimensions
        b, seq_len, _, h, w = input_tensor.size()

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            print(cur_layer_input.size())
            layer_output = self.conv_lstm_blocks[layer_idx](
                cur_layer_input=cur_layer_input,
                hidden_state=hidden_state[layer_idx])

            cur_layer_input = layer_output
            layer_output_list.append(layer_output)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]

        return layer_output_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            inv_scaling_factor = 2**i  # Down-sampling resulting from max-pooling
            cur_image_size = (int(image_size[0]/inv_scaling_factor), int(image_size[1]/inv_scaling_factor))
            print(f'Layer {i} image size: {cur_image_size}')
            init_states.append(self.conv_lstm_blocks[i].conv_lstm.init_hidden(batch_size, cur_image_size))
        print('\n')
        return init_states


class ConvLSTMBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.conv_lstm = ConvLSTMCell(input_dim, hidden_dim=hidden_dim,
                                      kernel_size=[kernel_size, kernel_size], bias=bias)
        self.mp2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(num_features=input_dim)

    def forward(self, cur_layer_input, hidden_state):
        # print('Inside block forward!')
        b, seq_len, channels, height, width = cur_layer_input.size()
        h, c = hidden_state
        output_inner = []
        for t in range(seq_len):
            h, c = self.conv_lstm(
                input_tensor=cur_layer_input[:, t, :, :, :],
                cur_state=[h, c])
            output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
        # print(layer_output.size())
        x = layer_output.view(b * seq_len, channels, height, width)
        x = self.mp2d(x)
        x = x.view(b, seq_len, channels, int(height/2), int(width/2))
        # print(x.size())
        # x = self.bn(x)
        # print(x.size())
        return x


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width),
                torch.zeros(batch_size, self.hidden_dim, height, width))


if __name__ == '__main__':

    trainer = pl.Trainer(fast_dev_run=True)
    # trainer.fit(conv_lstm)

    conv_lstm_model = ConvLSTMModel(input_channels=3, hidden_per_layer=[3, 3, 3],
                                    kernel_size_per_layer=[5, 5, 5])
    output_list = conv_lstm_model(torch.rand(5, 10, 3, 224, 224))
    print(len(output_list))
    print(output_list[0].size())
