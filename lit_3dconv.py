from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.model3D_1_224 import VGGStyle3DCNN
from lit_convlstm import ConvLSTMModule
import pytorch_lightning as pl
from torch import nn
import torchmetrics
import torch


class ThreeDCNNModule(ConvLSTMModule):
    def __init__(self, input_size, lr, reduce_lr, momentum, weight_decay, dropout):
        super(ConvLSTMModule, self).__init__()

        self.b, self.t, self.c, self.h, self.w = input_size
        self.seq_first = False
        self.lr = lr
        self.reduce_lr = reduce_lr
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.reduce_lr:
            scheduler = ReduceLROnPlateau(
                optimizer, 'max', factor=0.5, patience=2, verbose=True)
            return {'optimizer': optimizer,
                    'lr_scheduler': scheduler,
                    'monitor': 'val_acc'}
        else:
            return optimizer
