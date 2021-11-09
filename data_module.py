import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from transforms_video import ComposeMix, RandomRotationVideo, Scale, RandomCropVideo, RandomHorizontalFlipVideo
from data_loader_av import VideoFolder, UCFHMDBFullDataset


class Diving48DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, config: dict, seq_first: bool):
        super().__init__()
        self.data_dir = data_dir
        self.dims = (3, config['input_spatial_size'], config['input_spatial_size'])
        self.dims = (3, 224, 224)
        self.num_classes = 48
        self.upscale_size_train = config['upscale_size_train']
        self.upscale_size_eval = config['upscale_size_eval']
        self.config = config
        self.seq_first = seq_first

        self.transform_train_pre = ComposeMix([
            [RandomHorizontalFlipVideo(p=0.5), "vid"]
            # [RandomRotationVideo(15), "vid"],
            # [Scale(self.upscale_size_train), "img"],
            # [RandomCropVideo(self.dims[1]), "vid"],
        ])

        # Center crop videos during evaluation
        self.transform_eval_pre = ComposeMix([
           #  [Scale(self.upscale_size_eval), "img"],
           #  [torchvision.transforms.ToPILImage(), "img"],
           #  [torchvision.transforms.CenterCrop(self.dims[1]), "img"],
        ])

        # Transforms common to train and eval sets and applied after "pre" transforms
        self.transform_post = ComposeMix([
            [torchvision.transforms.ToTensor(), "img"],
            [torchvision.transforms.Resize((self.dims[1], self.dims[2])), "img"],
            [torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # default values for imagenet
                std=[0.229, 0.224, 0.225]), "img"]
        ])

    def setup(self, stage: str = None) -> None:
        # if stage == 'fit' or stage is None:
        if stage == 'fit':
            train_val_data = VideoFolder(
                root=self.data_dir,
                json_file_input=self.config['json_data_train'],
                json_file_labels=self.config['json_file_labels'],
                clip_size=self.config['clip_size'],
                nclips=self.config['nclips_train_val'],
                step_size=self.config['step_size_train_val'],
                is_val=False,
                frame_sample_mode=self.config['frame_sample_mode'],
                transform_pre=self.transform_train_pre,
                transform_post=self.transform_post,
                augmentation_mappings_json=self.config['augmentation_mappings_json'],
                augmentation_types_todo=self.config['augmentation_types_todo'],
                get_item_id=True,
                seq_first=self.seq_first,
                extension=self.config['training_extension']
                )
            self.train_data, self.val_data = random_split(
                train_val_data, [self.config['nb_train_samples'], self.config['nb_val_samples']],
                generator=torch.Generator().manual_seed(42))

        # if stage == 'test' or stage is None:
        if stage == 'test':
            self.test_data = VideoFolder(
                root=self.data_dir,
                json_file_input=self.config['json_data_test'],
                json_file_labels=self.config['json_file_labels'],
                clip_size=self.config['clip_size'],
                nclips=self.config['nclips_test'],
                step_size=self.config['step_size_test'],
                is_val=True,
                frame_sample_mode=self.config['frame_sample_mode'],
                transform_pre=self.transform_eval_pre,
                transform_post=self.transform_post,
                get_item_id=True,
                is_test=True,
                seq_first=self.seq_first,
                extension=self.config['test_extension']
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data, batch_size=self.config['batch_size'], shuffle=True,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=False)


class UCFHMDBFullDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, config: dict, seq_first: bool):
        super().__init__()
        self.data_dir = data_dir
        self.dims = (3, config['input_spatial_size'], config['input_spatial_size'])
        self.dims = (3, 112, 112)
        self.num_classes = 12
        self.upscale_size_train = config['upscale_size_train']
        self.upscale_size_eval = config['upscale_size_eval']
        self.config = config
        self.seq_first = seq_first

        self.transform_train_pre = ComposeMix([
            [RandomHorizontalFlipVideo(p=0.5), "vid"]
        ])

        # Center crop videos during evaluation
        self.transform_eval_pre = ComposeMix([
        ])

        # Transforms common to train and eval sets and applied after "pre" transforms
        self.transform_post = ComposeMix([
            [torchvision.transforms.ToTensor(), "img"],
            [torchvision.transforms.Resize((self.dims[1], self.dims[2])), "img"],
            [torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # default values for imagenet
                std=[0.229, 0.224, 0.225]), "img"]
        ])

    def setup(self, stage: str = None) -> None:
        # if stage == 'fit' or stage is None:
        if stage == 'fit':
            self.train_data = UCFHMDBFullDataset(
                root=self.data_dir,
                annotation_file=self.config['train_annotations'],
                clip_size=self.config['clip_size'],
                nclips=self.config['nclips_train_val'],
                step_size=self.config['step_size_train_val'],
                is_val=False,
                frame_sample_mode=self.config['frame_sample_mode'],
                transform_pre=self.transform_train_pre,
                transform_post=self.transform_post,
                augmentation_mappings_json=self.config['augmentation_mappings_json'],
                augmentation_types_todo=self.config['augmentation_types_todo'],
                get_item_id=True,
                seq_first=self.seq_first,
                extension=self.config['training_extension']
            )

            self.val_data = UCFHMDBFullDataset(
                root=self.data_dir,
                annotation_file=self.config['val_annotations'],
                clip_size=self.config['clip_size'],
                nclips=self.config['nclips_train_val'],
                step_size=self.config['step_size_train_val'],
                is_val=True,
                frame_sample_mode=self.config['frame_sample_mode'],
                transform_pre=self.transform_train_pre,
                transform_post=self.transform_post,
                augmentation_mappings_json=self.config['augmentation_mappings_json'],
                augmentation_types_todo=self.config['augmentation_types_todo'],
                get_item_id=True,
                seq_first=self.seq_first,
                extension=self.config['training_extension']
            )

        # if stage == 'test' or stage is None:
        if stage == 'test':
            self.test_data = UCFHMDBFullDataset(
                root=self.data_dir,
                annotation_file=self.config['test_annotations'],
                clip_size=self.config['clip_size'],
                nclips=self.config['nclips_test'],
                step_size=self.config['step_size_test'],
                is_val=True,
                frame_sample_mode=self.config['frame_sample_mode'],
                transform_pre=self.transform_eval_pre,
                transform_post=self.transform_post,
                get_item_id=True,
                seq_first=self.seq_first,
                extension=self.config['test_extension']
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data, batch_size=self.config['batch_size'], shuffle=True,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=False)
