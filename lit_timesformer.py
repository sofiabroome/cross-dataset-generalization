from timesformer_pytorch.timesformer_pytorch import attn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timesformer_pytorch import TimeSformer
from sklearn.metrics import classification_report
import pytorch_lightning as pl
from lit_convlstm import ConvLSTMModule

from torch import nn
import torchmetrics
import torch


class TimeSformerModule(ConvLSTMModule):
    def __init__(self, input_size, optimizer, nb_labels, lr, reduce_lr, dim,
                 patch_size, attn_dropout, ff_dropout, depth, heads, dim_head,
                 momentum, weight_decay, dropout_classifier):
                 
        super(ConvLSTMModule, self).__init__()
        self.b, self.t, self.c, self.h, self.w = input_size
        self.seq_first = True
        self.nb_labels = nb_labels
        self.optimizer = optimizer
        self.lr = lr
        self.reduce_lr = reduce_lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.dim = dim
        self.patch_size = patch_size
        self.num_frames = self.t
        self.depth = depth

        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        self.transformer_encoder = TimeSformer(
                dim = self.dim,  #?
                image_size = self.w if self.w <= self.h else self.h,  # NOTE: TimeSformer accepts square images
                patch_size = self.patch_size,
                num_frames = self.num_frames,
                num_classes = self.nb_labels,
                depth = self.depth,  #?
                heads = self.heads,  #?
                dim_head =  self.dim_head,  #?
                attn_dropout = self.attn_dropout,
                ff_dropout = self.ff_dropout,
                channels=self.c
            )

        self.softmax = nn.Softmax(dim=1)
        self.accuracy = torchmetrics.Accuracy()
        self.top5_accuracy = torchmetrics.Accuracy(top_k=5)
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=nb_labels)
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        x = self.transformer_encoder(x)
        return x

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(), self.lr, momentum=self.momentum,
                weight_decay=self.weight_decay)
        if self.optimizer == 'Adam':
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
