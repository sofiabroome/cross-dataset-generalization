from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.cnn3d import VGGStyle3DCNN
from lit_convlstm import ConvLSTMModule
from torch import nn
import torchmetrics
import torch


class ThreeDCNNModule(ConvLSTMModule):
    def __init__(self, input_size, optimizer, hidden_per_layer, kernel_size_per_layer,
                 conv_stride, pooling, dropout_encoder, nb_labels, lr, reduce_lr,
                 momentum, weight_decay, dropout_classifier):
        super(ConvLSTMModule, self).__init__()

        self.b, self.t, self.c, self.h, self.w = input_size
        self.seq_first = False
        self.out_features = nb_labels
        self.optimizer = optimizer
        self.lr = lr
        self.reduce_lr = reduce_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.hidden_per_layer = hidden_per_layer
        self.kernel_size_per_layer = kernel_size_per_layer
        self.conv_stride = conv_stride
        self.pooling = pooling
        self.dropout_encoder = dropout_encoder
        self.conv3d_encoder = VGGStyle3DCNN(input_channels=self.c, hidden_per_layer=hidden_per_layer,
                                            kernel_size_per_layer=kernel_size_per_layer,
                                            conv_stride=conv_stride, pooling=pooling, dropout=dropout_encoder)

        input_tensor = torch.autograd.Variable(torch.rand(1, self.c, self.t, self.h, self.w))
        sample_output = self.conv3d_encoder(input_tensor)
        self.encoder_out_dim = torch.prod(torch.tensor(sample_output.shape[1:]))

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.dropout = nn.Dropout(p=dropout_classifier)
        self.linear = nn.Linear(
            in_features=self.encoder_out_dim,
            out_features=self.out_features)
        self.accuracy = torchmetrics.Accuracy()
        self.top5_accuracy = torchmetrics.Accuracy(top_k=5)
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=nb_labels)
        self.softmax = nn.Softmax(dim=1)
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        x = self.conv3d_encoder(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def configure_optimizers(self):
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
