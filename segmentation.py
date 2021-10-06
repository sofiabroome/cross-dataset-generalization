from frames_for_segmentation_dataset import VideoFramesForSegmentation
from torchvision.utils import draw_segmentation_masks
from segmentation_test_script import plot_with_mask
import torchvision.transforms.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import torch
import utils


def main():
    # load configurations

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='json config file path')
    args = parser.parse_args()
    config = utils.load_json_config(args.config)

    model = torch.hub.load('pytorch/vision:v0.10.0', config['deeplab_backbone'], pretrained=True)
    model.eval()

    loader = VideoFramesForSegmentation(
        root=config['data_folder'],
        json_file_input=config['json_data_train'],
        json_file_labels=config['json_file_labels'],
        clip_size=config['clip_size'],
        nclips=config['nclips_train_val'],
        step_size=config['step_size_train_val'],
        is_val=False)



if __name__ == '__main__':
    main()
