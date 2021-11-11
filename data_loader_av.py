import av
import time

import pandas as pd
import torch
import numpy as np

from data_parser import WebmDataset, CroppedMp4Dataset
from data_augmentor import Augmentor
import torchvision
from transforms_video import *
from utils import save_images_for_debug
import itertools
import os
# from pytorchvideo.pytorchvideo.data.encoded_video_pyav import EncodedVideo
from pytorchvideo.data.encoded_video_pyav import EncodedVideo


FRAMERATE = 1  # default value


class VideoFolder(torch.utils.data.Dataset):

    def __init__(self, root, json_file_input, json_file_labels, clip_size,
                 nclips, step_size, is_val, frame_sample_mode,
                 transform_pre=None, transform_post=None,
                 augmentation_mappings_json=None, augmentation_types_todo=None,
                 get_item_id=False, is_test=False, seq_first=False,
                 extension='.mp4'):
        if extension == '.webm':
            self.dataset_object = WebmDataset(json_file_input, json_file_labels,
                                              root, is_test=is_test)
        if extension == '_cropped.mp4':
            self.dataset_object = CroppedMp4Dataset(
                json_file_input, json_file_labels, root, is_test=is_test)

        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.frame_sample_mode = frame_sample_mode
        self.transform_pre = transform_pre
        self.transform_post = transform_post

        self.augmentor = Augmentor(augmentation_mappings_json,
                                   augmentation_types_todo)

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val
        self.get_item_id = get_item_id
        self.seq_first = seq_first

    def __getitem__(self, index):

        item = self.json_data[index]

        # Open video file
        reader = av.open(item.path)  # Takes around 0.005 seconds.

        try:
            imgs = []
            imgs = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]  # 0.5 s.
        except (RuntimeError, ZeroDivisionError) as exception:
            print('{}: WEBM reader cannot open {}. Empty '
                  'list returned.'.format(type(exception).__name__, item.path))

        if self.frame_sample_mode == 'evenly_distributed':
            evenly_spaced_frame_indices = np.round(
                np.linspace(0, len(imgs) - 1, self.clip_size)
            ).astype(int)
            imgs = [imgs[ind] for ind in evenly_spaced_frame_indices]
        # print('befor pre', type(imgs[0]))
        imgs = self.transform_pre(imgs)
        # print('after pre', type(imgs[0]))
        imgs, label = self.augmentor(imgs, item.label)
        # print('after aug', type(imgs[0]))
        imgs = self.transform_post(imgs)
        # print('after post', type(imgs[0]))

        num_frames = len(imgs)

        if self.frame_sample_mode == 'augmented':
            if self.nclips > -1:
                num_frames_necessary = self.clip_size * self.nclips * self.step_size
            else:
                num_frames_necessary = num_frames
            offset = 0
            if num_frames_necessary < num_frames:
                # If there are more frames, then sample starting offset.
                diff = (num_frames - num_frames_necessary)
                # temporal augmentation
                if not self.is_val:
                    offset = np.random.randint(0, diff)

            if len(imgs) < (self.clip_size * self.nclips):
                imgs.extend([imgs[-1]] *
                            ((self.clip_size * self.nclips) - len(imgs)))

            imgs = imgs[offset: num_frames_necessary + offset: self.step_size]

        if 'something' in self.root:
            target_idx = self.classes_dict[label]
        else:
            target_idx = label

        # format data to torch
        data = torch.stack(imgs)
        if not self.seq_first:
            data = data.permute(1, 0, 2, 3)
        if self.get_item_id:
            return data, target_idx, item.id
        else:
            return data, target_idx

    def __len__(self):
        return len(self.json_data)


class UCFHMDBFullDataset(torch.utils.data.Dataset):

    def __init__(self, root, annotation_file, clip_size,
                 nclips, step_size, is_val, frame_sample_mode,
                 transform_pre=None, transform_post=None,
                 augmentation_mappings_json=None, augmentation_types_todo=None,
                 get_item_id=False, seq_first=False, extension='.avi'):
        self.annotation_df = pd.read_csv(annotation_file, delim_whitespace=True)
        self.root = root
        self.frame_sample_mode = frame_sample_mode
        self.transform_pre = transform_pre
        self.transform_post = transform_post

        self.augmentor = Augmentor(augmentation_mappings_json,
                                   augmentation_types_todo)

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val
        self.get_item_id = get_item_id
        self.seq_first = seq_first
        self.extension = extension

    def __getitem__(self, index):

        item = self.annotation_df.iloc[index]
        item_path = os.path.join(self.root, item['video_id'] + self.extension)

        # # print(item_path)
        # stream = 'video'
        # video_object = torchvision.io.VideoReader(item_path, stream)
        # # to_pil = torchvision.transforms.ToPILImage()
        # imgs = []
        end = float(item['nb_frames']/30)
        # for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(0)):
        #     imgs.append(frame['data'])
        #     # imgs.append(frame['data'].numpy())

        # Open video file
        # reader = av.open(item_path)  # Takes around 0.005 seconds.
        video = EncodedVideo.from_path(item_path)
        imgs = video.get_clip(start_sec=0, end_sec=end)
        # print('Keys:', imgs.keys())
        imgs = imgs['video']

        if imgs == None:
            print(item['nb_frames'])
            print(end)
            print(item_path)
            print(index)
        # print(imgs.shape)
        # print(imgs[0].shape)
        # print(imgs[4].shape)

        # try:
        #     imgs = []
        #     imgs = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]  # 0.5 s.
        # except (RuntimeError, ZeroDivisionError) as exception:
        #     print('{}: The AV reader cannot open {}. Empty '
        #           'list returned.'.format(type(exception).__name__, item_path))

        if self.frame_sample_mode == 'evenly_distributed':
            evenly_spaced_frame_indices = np.round(
                np.linspace(0, len(imgs) - 1, self.clip_size)).astype(int)
            imgs = [imgs[:,ind,:,:] for ind in evenly_spaced_frame_indices]
            # imgs = [np.moveaxis(imgs[ind], 0, -1) for ind in evenly_spaced_frame_indices]

        # print('after: ', len(imgs))
        # print(imgs[0].shape)
        imgs = self.transform_pre(imgs)
        # imgs = [flip(imgs[ind]) for ind in range(self.clip_size)]
        # print('got past pre')
        # print(item['class'])
        # imgs, label = self.augmentor(imgs, item['class'])
        # print('label after', label)
        label = item['class']
        # print('got past aug')
        imgs = [torch.div(imgs[ind], 255.) for ind in range(self.clip_size)]
        # print(imgs[0])
        # imgs = self.transform_post(imgs)
        # print('got past post')

        resize = torchvision.transforms.Resize((112, 112))
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        imgs = [resize(imgs[ind]) for ind in range(self.clip_size)]

        num_frames = len(imgs)

        if self.frame_sample_mode == 'augmented':
            if self.nclips > -1:
                num_frames_necessary = self.clip_size * self.nclips * self.step_size
            else:
                num_frames_necessary = num_frames
            offset = 0
            if num_frames_necessary < num_frames:
                # If there are more frames, then sample starting offset.
                diff = (num_frames - num_frames_necessary)
                # temporal augmentation
                if not self.is_val:
                    offset = np.random.randint(0, diff)

            if len(imgs) < (self.clip_size * self.nclips):
                imgs.extend([imgs[-1]] *
                            ((self.clip_size * self.nclips) - len(imgs)))

            imgs = imgs[offset: num_frames_necessary + offset: self.step_size]

        target_idx = label

        # format data to torch
        data = torch.stack(imgs)
        if not self.seq_first:
            data = data.permute(1, 0, 2, 3)
        if self.get_item_id:
            return data, target_idx, item['video_id']
        else:
            return data, target_idx

    def __len__(self):
        return len(self.annotation_df)


if __name__ == '__main__':
    upscale_size = int(84 * 1.1)
    transform_pre = ComposeMix([
            # [RandomRotationVideo(20), "vid"],
            [Scale(upscale_size), "img"],
            [RandomCropVideo(84), "vid"],
            # [RandomHorizontalFlipVideo(0), "vid"],
            # [RandomReverseTimeVideo(1), "vid"],
            # [torchvision.transforms.ToTensor(), "img"],
             ])
    # identity transform
    transform_post = ComposeMix([
                        [torchvision.transforms.ToTensor(), "img"],
                         ])

    smth_root = '/local_storage/users/sbroome/something-something/20bn-something-something-v2/'

    loader = VideoFolder(root=smth_root,
                         json_file_input=smth_root + "annotations/something-something-v2-train.json",
                         json_file_labels=smth_root + "annotations/something-something-v2-labels.json",
                         clip_size=36,
                         nclips=1,
                         step_size=1,
                         is_val=False,
                         frame_sample_mode='evenly_distributed',
                         transform_pre=transform_pre,
                         transform_post=transform_post,
                         # augmentation_mappings_json="notebooks/augmentation_mappings.json",
                         # augmentation_types_todo=["left/right", "left/right agnostic", "jitter_fps"],
                         )
    # fetch a sample
    # data_item, target_idx = loader[1]
    # save_images_for_debug("input_images_2", data_item.unsqueeze(0))
    # print("Label = {}".format(loader.classes_dict[target_idx]))

    import time
    from tqdm import tqdm

    batch_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=False,
        num_workers=2, pin_memory=True)

    start = time.time()
    for i, a in enumerate(tqdm(batch_loader)):
        if i > 100:
            break
        pass
    print("Size --> {}".format(a[0].size()))
    print(time.time() - start)
