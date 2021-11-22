import os
import json
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_json_config(path):
    """ loads a json config file"""
    with open(path) as data_file:
        config = json.load(data_file)
        config = config_init(config)
    return config


def config_init(config):
    """ Some of the variables that should exist and contain default values """
    if "augmentation_mappings_json" not in config:
        config["augmentation_mappings_json"] = None
    if "augmentation_types_todo" not in config:
        config["augmentation_types_todo"] = None
    return config


def save_images_for_debug(dir_img, imgs):
    """
    2x3x12x224x224 --> [BS, C, seq_len, H, W]
    """
    print("Saving images to {}".format(dir_img))
    from matplotlib import pylab as plt
    imgs = imgs.permute(0, 2, 3, 4, 1)  # [BS, seq_len, H, W, C]
    imgs = imgs.mul(255).numpy()
    if not os.path.exists(dir_img):
        os.makedirs(dir_img)
    print(imgs.shape)
    for batch_id, batch in enumerate(imgs):
        batch_dir = os.path.join(dir_img, "batch{}".format(batch_id + 1))
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        for j, img in enumerate(batch):
            plt.imsave(os.path.join(batch_dir, "frame{%04d}.png" % (j + 1)),
                       img.astype("uint8"))


def plot_class_histogram(dir_img, all_targets_list, which_split, nb_classes=48):
    print("Saving histogram to {}".format(dir_img))
    from matplotlib import pyplot as plt
    print('Computing hist...')
    plt.hist(torch.cat(all_targets_list), bins=nb_classes, range=(0, 47))

    print('Finished computing hist. Saving...')
    if not os.path.exists(dir_img):
        os.makedirs(dir_img)

    plt.savefig(os.path.join(dir_img, f"{which_split}_histogram.png"))


