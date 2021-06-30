import torch
import torchvision
from torch import nn
import torch.functional as F
import pytorch_lightning as pl

import utils
from transforms_video import *
from data_loader_av import VideoFolder
from models.convlstm import StackedConvLSTMModel


# load configurations
args = utils.load_args()
config = utils.load_json_config(args.config)


class ConvLSTMModule(pl.LightningModule):
    def __init__(self, input_size, hidden_per_layer, kernel_size_per_layer):
        super(ConvLSTMModule, self).__init__()

        self.b, self.t, self.c, self.h, self.w = input_size
        self.seq_first = False
        self.num_layers = len(hidden_per_layer)
        self.convlstm_encoder = StackedConvLSTMModel(
            3, hidden_per_layer, kernel_size_per_layer)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(
            in_features=self.t * self.c * int(self.h / 2**self.num_layers) * int(self.w/2**self.num_layers),
            out_features=2)

    def forward(self, x) -> torch.Tensor:
        x = self.convlstm_encoder(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, item_id = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def train_dataloader(self):
        upscale_size_train = int(config['input_spatial_size'] * config["upscale_factor_train"])
        transform_train_pre = ComposeMix([
            [RandomRotationVideo(15), "vid"],
            [Scale(upscale_size_train), "img"],
            [RandomCropVideo(config['input_spatial_size']), "vid"],
        ])

        # Transforms common to train and eval sets and applied after "pre" transforms
        transform_post = ComposeMix([
            [torchvision.transforms.ToTensor(), "img"],
            [torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # default values for imagenet
                std=[0.229, 0.224, 0.225]), "img"]
        ])

        train_val_data = VideoFolder(root=config['data_folder'],
                                     json_file_input=config['json_data_train'],
                                     json_file_labels=config['json_file_labels'],
                                     clip_size=config['clip_size'],
                                     nclips=config['nclips_train_val'],
                                     step_size=config['step_size_train_val'],
                                     is_val=False,
                                     transform_pre=transform_train_pre,
                                     transform_post=transform_post,
                                     augmentation_mappings_json=config['augmentation_mappings_json'],
                                     augmentation_types_todo=config['augmentation_types_todo'],
                                     get_item_id=True,
                                     seq_first=self.seq_first
                                     )
        train_data, val_data = torch.utils.data.random_split(
            train_val_data, [config['nb_train_samples'], config['nb_val_samples']],
            generator=torch.Generator().manual_seed(42))

        print(" > Using {} processes for data loader.".format(
            config["num_workers"]))

        return torch.utils.data.DataLoader(
            train_data,
            batch_size=config['batch_size'], shuffle=True,
            num_workers=config['num_workers'], pin_memory=True,
            drop_last=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), config['lr'],
                               momentum=config['momentum'],
                               weight_decay=config['weight_decay'])


if __name__ == '__main__':
    trainer = pl.Trainer(fast_dev_run=True)
    input_tensor = torch.rand(5, 10, 3, 224, 224)
    convlstm = ConvLSTMModule(input_size=input_tensor.size(), hidden_per_layer=[3, 3, 3],
                              kernel_size_per_layer=[5, 5, 5])
    trainer.fit(convlstm)
    print(convlstm(input_tensor))