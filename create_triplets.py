import glob
from PIL import Image
import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
from os import walk
import random
import re

family_np_folder = 'single_images'

df = pd.read_csv('train_relationships.csv')


def create_triples():
    triples = []
    for _, row in df.iterrows():
        family_members = os.listdir(family_np_folder)
        anchor = f'{family_np_folder}/{row.p1}.npy'
        pos = f'{family_np_folder}/{row.p2}.npy'
        current_family = row.p1.split('_')[0]
        random_member = np.random.choice(family_members)
        while random_member.split('_')[0] == current_family:
            random_member = np.random.choice(family_members)
        neg = f'{family_np_folder}/{random_member}'
        triples.append([anchor, pos, neg])
    return triples


def assert_if_triple_exists(triples):
    indices_to_be_removed = []
    for index, triple in enumerate(triples):
        for member in triple:
            if not os.path.exists(member):
                indices_to_be_removed.append(index)

    for remove_index in sorted(indices_to_be_removed, reverse=True):
        del triples[remove_index]
        print(f'Removing triple : triple_number_{remove_index}')
    return triples


def create_doubles_from_triples(triples):
    doubles = []
    for anchor, pos, neg in triples:
        doubles.append([anchor, pos, 0])
        doubles.append([anchor, neg, 1])
    return doubles

if __name__ == "__main__":
    triples = create_triples()
    triples = assert_if_triple_exists(triples)
    doubles = create_doubles_from_triples(triples)
    np.save('triples.npy', np.array(triples))
    np.save('doublets.npy', np.array(doubles))
