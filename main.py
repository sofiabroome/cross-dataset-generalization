import os
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

import utils
import argparse
import torchmetrics
from data_module import Diving48DataModule
from models.convlstm import StackedConvLSTMModel


class ConvLSTMModule(pl.LightningModule):
    def __init__(self, input_size, hidden_per_layer, kernel_size_per_layer, conv_stride):
        super(ConvLSTMModule, self).__init__()

        self.b, self.t, self.c, self.h, self.w = input_size
        self.seq_first = True
        self.num_layers = len(hidden_per_layer)
        self.convlstm_encoder = StackedConvLSTMModel(
            self.c, hidden_per_layer, kernel_size_per_layer, conv_stride)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(
            in_features=self.t * hidden_per_layer[-1] * int(self.h /(2**self.num_layers*conv_stride)) *
            int(self.w/(2**self.num_layers*conv_stride)), out_features=48)
        self.accuracy = torchmetrics.Accuracy()
        self.save_hyperparameters() 

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
    # load configurations

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='json config file path')
    parser.add_argument('--eval_only', '-e', action='store_true', 
                        help="evaluate trained model on validation data.")
    parser.add_argument('--resume', '-r', action='store_true',
                        help="resume training from a given checkpoint.")
    parser.add_argument('--test_run', action='store_true',
                        help="quick test run")
    parser.add_argument('--job_identifier', '-j', help='Unique identifier for run,'
                                                       'avoids overwriting model.')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    config = utils.load_json_config(args.config)

    wandb_logger = WandbLogger(project='cross-dataset-generalization', config=config)

    seed_everything(42, workers=True)

    conv_lstm = ConvLSTMModule(input_size=(config['batch_size'], config['clip_size'], 3,
                               config['input_spatial_size'], config['input_spatial_size']),
                               hidden_per_layer=config['hidden_per_layer'],
                               kernel_size_per_layer=config['kernel_size_per_layer'],
                               conv_stride=config['conv_stride'])

    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',
                                          verbose=True,
                                          filename='{epoch}-{val_loss:.2f}-{val_acc:.4f}')


    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=2,
        progress_bar_refresh_rate=1,
        callbacks=[checkpoint_callback],
        weights_save_path=os.path.join(config['output_dir'], args.job_identifier),
        logger=wandb_logger)

    dm = Diving48DataModule(data_dir=config['data_folder'], config=config)

    trainer.fit(conv_lstm, dm)

    # trainer.test(conv_lstm, datamodule=dm)
    # trainer.test(datamodule=dm, model=conv_lstm,
    #              ckpt_path='xdataset_output/None/epoch=1-val_loss=3.47-val_acc=0.07.ckpt')

    
