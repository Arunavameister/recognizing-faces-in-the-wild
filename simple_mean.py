import glob
from PIL import Image
import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
import pandas as pd

from resnet50_ft_dag import Resnet50_ft_dag

weights_path = 'resnet50_ft_dag.pth'


def resnet50_ft_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Resnet50_ft_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model


model = resnet50_ft_dag(weights_path=weights_path)

width = 224
height = 224
target_channels = 3

save_folder = 'families'


def simple_mean(folder):
    for family in glob.iglob(f'{folder}/*'):
        for members in glob.iglob(f'{family}/*'):
            _, _, num_files = next(os.walk(members))
            # print(len(num_files))
            members_img_array = np.zeros((len(num_files), width, height, target_channels))
            for i, photos in enumerate(glob.iglob(f'{members}/*.jpg')):
                img = Image.open(photos).convert('RGB')
                img = img.resize((width, height))
                img = np.array(img) / 255.
                members_img_array[i] = img
                np.save(f'{members}/members_img_array.npy', members_img_array)

            if len(os.listdir(f'{members}')) == 0:
                print(f'Empty directory : {members}')
                continue
            member_mean = np.mean(np.load(f'{members}/members_img_array.npy'), axis=0)
            #member_mean = np.expand_dims(member_mean, axis=2)
            # result_member_mean = np.zeros((width, height))
            # result_member_mean = result_member_mean[:member_mean[:int(resnet_output / 2)],
            #                      :member_mean[int(resnet_output / 2)]:(resnet_output // 2) + (resnet_output % 2 > 0)]
            family_name = family.split('/')[1]
            member_name = members.split('/')[2]
            save_path = f'{save_folder}/{family_name}_{member_name}'
            np.save(f'{save_path}.npy', member_mean)

if __name__ == "__main__":
    simple_mean('train')
