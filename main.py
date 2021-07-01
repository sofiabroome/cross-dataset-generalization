import torch
from torch import nn
import pytorch_lightning as pl

import utils
from data_module import Diving48DataModule
from models.convlstm import StackedConvLSTMModel


# load configurations
args = utils.load_args()
config = utils.load_json_config(args.config)


class ConvLSTMModule(pl.LightningModule):
    def __init__(self, input_size, hidden_per_layer, kernel_size_per_layer):
        super(ConvLSTMModule, self).__init__()

        self.b, self.t, self.c, self.h, self.w = input_size
        self.seq_first = True
        self.num_layers = len(hidden_per_layer)
        self.convlstm_encoder = StackedConvLSTMModel(
            3, hidden_per_layer, kernel_size_per_layer)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(
            in_features=self.t * self.c * int(self.h / 2**self.num_layers) * int(self.w/2**self.num_layers),
            out_features=48)

    def forward(self, x) -> torch.Tensor:
        x = self.convlstm_encoder(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def loss_function(self, y_hat, y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y, item_id = batch
        y_hat = self(x)
        return {'loss': self.loss_function(y_hat, y)}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), config['lr'],
                               momentum=config['momentum'],
                               weight_decay=config['weight_decay'])


if __name__ == '__main__':
    trainer = pl.Trainer(fast_dev_run=True)
    dm = Diving48DataModule(data_dir=config['data_folder'], config=config)
    # trainer = pl.Trainer(max_epochs=config['num_epochs'],
    #                      progress_bar_refresh_rate=1)
    input_tensor = torch.rand(5, 6, 3, 224, 224)
    conv_lstm = ConvLSTMModule(input_size=input_tensor.size(), hidden_per_layer=[3, 3, 3],
                              kernel_size_per_layer=[5, 5, 5])
    trainer.fit(conv_lstm, dm)
