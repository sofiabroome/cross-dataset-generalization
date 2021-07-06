from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.model3D_1_224 import VGGStyle3DCNN
import pytorch_lightning as pl
from torch import nn
import torchmetrics
import torch


class ThreeDCNNModule(pl.LightningModule):
    def __init__(self, input_size, lr, momentum, weight_decay, dropout):
        super(ThreeDCNNModule, self).__init__()

        self.b, self.t, self.c, self.h, self.w = input_size
        self.seq_first = False
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.threed_cnn_encoder = VGGStyle3DCNN()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(
            in_features=512,
            # in_features=int(self.t/2) * 512 * 7 * 7,
            out_features=48)
        self.accuracy = torchmetrics.Accuracy()
        self.top5_accuracy = torchmetrics.Accuracy(top_k=5)
        self.softmax = nn.Softmax(dim=1)
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        x = self.threed_cnn_encoder(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    @staticmethod
    def loss_function(y_hat, y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y, item_id = batch
        y_hat = self(x)
        loss, acc, _ = self.get_loss_acc(y_hat, y)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, item_id = batch
        y_hat = self(x)
        loss, acc, top5_acc = self.get_loss_acc(y_hat, y)
        # By default, on_step=False, on_epoch=True for log calls in val and test
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_top5_acc', top5_acc, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y, item_id = batch
        y_hat = self(x)
        loss, acc, top5_acc = self.get_loss_acc(y_hat, y)
        self.log('test_acc', acc, prog_bar=True, sync_dist=True)
        self.log('test_top5_acc', top5_acc, sync_dist=True)
        self.log('test_loss', loss, sync_dist=True)
        return {'test_loss': loss}

    def get_loss_acc(self, y_hat, y):
        loss = self.loss_function(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        top5_acc = self.top5_accuracy(self.softmax(y_hat), y)
        return loss, acc, top5_acc

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer, 'max', factor=0.5, patience=2, verbose=True)
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_acc'}

