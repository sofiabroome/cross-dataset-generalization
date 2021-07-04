import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

import utils
import torchmetrics
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
            in_features=self.t * hidden_per_layer[-1] * int(self.h / 2**self.num_layers) *
            int(self.w/2**self.num_layers), out_features=48)
        self.accuracy = torchmetrics.Accuracy()

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
        loss, acc = self.get_loss_acc(y_hat, y)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, item_id = batch
        y_hat = self(x)
        loss, acc = self.get_loss_acc(y_hat, y)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_loss', loss)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y, item_id = batch
        y_hat = self(x)
        loss, acc = self.get_loss_acc(y_hat, y)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_loss', loss)
        return {'test_loss': loss}

    def get_loss_acc(self, y_hat, y):
        loss = self.loss_function(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        return loss, acc

    # def validation_epoch_end(self, outputs) -> None:
    #     val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
    #     log = {'avg_val_loss': val_loss}
    #     return {'log': log, 'val_loss': val_loss, 'val_acc': val_acc}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), config['lr'],
                               momentum=config['momentum'],
                               weight_decay=config['weight_decay'])


if __name__ == '__main__':
    wandb_logger = WandbLogger(project='cross-dataset-generalization', config=config)
    # trainer = pl.Trainer(fast_dev_run=True, gpus=1, accelerator='ddp')
    # trainer = pl.Trainer(fast_dev_run=True, gpus=1)
    trainer = pl.Trainer(fast_dev_run=True)
    # seed_everything(42, workers=True)
    # trainer = pl.Trainer()
    dm = Diving48DataModule(data_dir=config['data_folder'], config=config)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', verbose=True,
                                          dirpath='xdataset_output/',
                                          filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}')
    # trainer = pl.Trainer(max_epochs=config['num_epochs'], gpus=1,
    #                    progress_bar_refresh_rate=1,
    #                    callbacks=[checkpoint_callback],
    #                    weights_save_path='xdataset_output/',
    #                    logger=wandb_logger)
    conv_lstm = ConvLSTMModule(input_size=(5, 6, 3, 224, 224), hidden_per_layer=[12, 12, 12],
                              kernel_size_per_layer=[5, 5, 5])
    trainer.fit(conv_lstm, dm)
    # trainer.test(conv_lstm, datamodule=dm)
    # trainer.test(datamodule=dm,
    #              ckpt_path='xdataset_output/cross-dataset-generalization/1gtp4f1a/checkpoints/epoch=0-val_loss=3.30-val_acc=0.10.ckpt')

    
