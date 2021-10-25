import os
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import seed_everything

import utils
import argparse
from data_module import Diving48DataModule
from lit_convlstm import ConvLSTMModule
from lit_3dconv import ThreeDCNNModule
from models.model_utils import count_parameters


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
        model = ConvLSTMModule(input_size=(config['batch_size'], config['clip_size'], 3,
                                           config['input_spatial_size'], config['input_spatial_size']),
                               optimizer=config['optimizer'],
                               nb_labels=config['nb_labels'],
                               hidden_per_layer=config['hidden_per_layer'],
                               kernel_size_per_layer=config['kernel_size_per_layer'],
                               conv_stride=config['conv_stride'],
                               lr=config['lr'], reduce_lr=config['reduce_lr'],
                               momentum=config['momentum'], weight_decay=config['weight_decay'],
                               dropout_classifier=config['dropout_classifier'],
                               return_sequence=config['return_sequence'],
                               if_not_sequence=config['if_not_sequence'])


    if config['model_name'] == 'lit_3dconv':
        model = ThreeDCNNModule(input_size=(config['batch_size'], config['clip_size'], 3,
                                            config['input_spatial_size'], config['input_spatial_size']),
                                optimizer=config['optimizer'],
                                hidden_per_layer=config['hidden_per_layer'],
                                kernel_size_per_layer=config['kernel_size_per_layer'],
                                conv_stride=config['conv_stride'],
                                dropout_encoder=config['dropout_encoder'],
                                pooling=config['pooling'],
                                nb_labels=config['nb_labels'],
                                lr=config['lr'], reduce_lr=config['reduce_lr'],
                                momentum=config['momentum'], weight_decay=config['weight_decay'],
                                dropout_classifier=config['dropout_classifier'])

    config['nb_encoder_params'], config['nb_trainable_params'] = count_parameters(model)
    print('\n Nb encoder params: ', config['nb_encoder_params'], 'Nb params total: ', config['nb_trainable_params'])

    checkpoint_callback = ModelCheckpoint(verbose=True,
                                          filename='{epoch}-{val_loss:.2f}-{val_acc:.4f}',
                                          every_n_epochs=1,
                                          save_top_k=-1)
    # checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',
    #                                       verbose=True,
    #                                       filename='{epoch}-{val_loss:.2f}-{val_acc:.4f}')

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=config['early_stopping_patience'],
        verbose=False,
        mode='max'
    )

    callbacks = [checkpoint_callback, early_stop_callback]
    
    if config['reduce_lr']:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=config['num_epochs'],
        progress_bar_refresh_rate=1,
        callbacks=callbacks,
        weights_save_path=os.path.join(config['output_dir'], args.job_identifier),
        logger=wandb_logger,
        plugins=DDPPlugin(find_unused_parameters=False))

    if trainer.gpus is not None:
        config['num_workers'] = int(trainer.gpus/8 * 128)
    else:
        config['num_workers'] = 0

    test_dm = Diving48DataModule(data_dir=config['test_data_folder'], config=config, seq_first=model.seq_first)
    test_dm_2 = Diving48DataModule(data_dir=config['test_data_folder_2'], config=config, seq_first=model.seq_first)

    if config['inference_from_checkpoint_only']:
        model_from_checkpoint = ConvLSTMModule.load_from_checkpoint(config['ckpt_path'])
        # trainer.test(datamodule=test_dm, model=model_from_checkpoint)

    else:
        train_dm = Diving48DataModule(data_dir=config['data_folder'], config=config, seq_first=model.seq_first)
        trainer.fit(model, train_dm)
        wandb_logger.log_metrics({'best_val_acc': trainer.checkpoint_callback.best_model_score})
        trainer.test(datamodule=test_dm)
        trainer.test(datamodule=test_dm_2)


if __name__ == '__main__':
    main()
