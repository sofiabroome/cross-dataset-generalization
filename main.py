import os
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import seed_everything

import utils
import argparse
from data_module import Diving48DataModule
from lit_convlstm import ConvLSTMModule


def main():
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

    if config['model_name'] == 'lit_convlstm':
        conv_lstm = ConvLSTMModule(input_size=(config['batch_size'], config['clip_size'], 3,
                                               config['input_spatial_size'], config['input_spatial_size']),
                                   hidden_per_layer=config['hidden_per_layer'],
                                   kernel_size_per_layer=config['kernel_size_per_layer'],
                                   conv_stride=config['conv_stride'],
                                   lr=config['lr'], momentum=config['momentum'],
                                   weight_decay=config['weight_decay'], dropout=config['dropout'])

    if config['model_name'] == 'lit_3dconv':
        pass

    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',
                                          verbose=True,
                                          filename='{epoch}-{val_loss:.2f}-{val_acc:.4f}')

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=config['early_stopping_patience'],
        verbose=False,
        mode='max'
    )

    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=config['num_epochs'],
        progress_bar_refresh_rate=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        weights_save_path=os.path.join(config['output_dir'], args.job_identifier),
        logger=wandb_logger,
        plugins=DDPPlugin(find_unused_parameters=False))

    if trainer.gpus is not None:
        config['num_workers'] = int(trainer.gpus/8 * 128)

    dm = Diving48DataModule(data_dir=config['data_folder'], config=config)

    trainer.fit(conv_lstm, dm)

    # trainer.test(conv_lstm, datamodule=dm)
    # trainer.test(datamodule=dm, model=conv_lstm,
    #              ckpt_path=config['ckpt_path'])


if __name__ == '__main__':
    main()
