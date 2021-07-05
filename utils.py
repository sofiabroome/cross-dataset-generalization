import os
import sys
import json
import pickle
import argparse
import torch
import shutil
import glob
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_evaluation_as_text(evaluation, str_name, config):
    f = open(config['save_dir'] + f'{str_name}.txt', 'w')
    print(evaluation, end="", file=f)
    f.close()
    print(evaluation)


def remove_module_from_checkpoint_state_dict(state_dict):
    """
    Removes the prefix `module` from weight names that gets added by
    torch.nn.DataParallel()
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


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


def setup_cuda_devices(args):
    device_ids = []
    device = torch.device("cuda" if args.use_cuda else "cpu")
    if device.type == "cuda":
        device_ids = [int(i) for i in args.gpus.split(',')]
    return device, device_ids


def save_checkpoint(state, is_best, config, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(config['output_dir'], config['model_id'], filename)
    model_path = os.path.join(config['output_dir'], config['model_id'], 'model_best.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        print(" > Best model found at this epoch. Saving ...")
        shutil.copyfile(checkpoint_path, model_path)


def save_results(logits_matrix, features_matrix, targets_list, item_id_list,
                 class_to_idx, config):
    """
    Saves the predicted logits matrix, true labels, sample ids and class
    dictionary for further analysis of results
    """
    print("Saving inference results ...")
    path_to_save = os.path.join(
        config['output_dir'], config['model_name'], "test_results.pkl")
    with open(path_to_save, "wb") as f:
        pickle.dump([logits_matrix, features_matrix, targets_list,
                     item_id_list, class_to_idx], f)


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


def get_submission(logits_matrix, item_id_list, class_to_idx, config):
    top5_classes_pred_list = []

    for i, id in enumerate(item_id_list):
        logits_sample = logits_matrix[i]
        logits_sample_top5  = logits_sample.argsort()[-5:][::-1]
        # top1_class_index = logits_sample.argmax()
        # top1_class_label = class_to_idx[top1_class_index]

        top5_classes_pred_list.append(logits_sample_top5)

    path_to_save = os.path.join(
            config['output_dir'], config['model_name'], "test_submission.csv")
    with open(path_to_save, 'w') as fw:
        for id, top5_pred in zip(item_id_list, top5_classes_pred_list):
            fw.write("{}".format(id))
            for elem in top5_pred:
                fw.write(";{}".format(elem))
            fw.write("\n")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ExperimentalRunCleaner(object):
    """
    Remove the output dir, if you exit with Ctrl+C and if there are less
    then 1 file. It prevents the noise of experimental runs.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def __call__(self, signal, frame):
        num_files = len(glob.glob(self.save_dir + "/*"))
        if num_files < 1:
            print('Removing: {}'.format(self.save_dir))
            shutil.rmtree(self.save_dir)
        print('You pressed Ctrl+C!')
        sys.exit(0)
