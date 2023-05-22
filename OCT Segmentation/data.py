import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import re
import os

from skimage.io import imread

import torch
import torch.utils.data


# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


# Turn a line into a <line_length x 1 x n_intensity values>, 0-255
# or an array of one-hot letter vectors
def lineToTensor(line):
    needed = 0


def train_test_split(*arrays, **options):

    # check if there are two arrays images and masks
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    max_size = options.pop('max_size')
    random_state = options.pop('random_state', None)
    split = options.pop('split', None)
    stratify = options.pop('stratify', None)
    shuffle = options.pop('shuffle', True)
    dataset = options.pop('dataset', None)
    fix_names = options.pop('fixnames', False)  # will fix the names from 99 to 099 to get the right order

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    n_samples = len(arrays[0])

    unique_patient = []
    positions = []
    # find all unique volumes and split by 'patient' save positions of each patient start
    for img in range(n_samples):
        if not unique_patient:
            get_name = arrays[0][img]
            get_name = get_name.replace('input\\' + dataset + '_images\\', '')
            get_num = get_name.split('_')[1]
            unique_patient.append(get_num)
            positions.append(img)

        else:
            get_name = arrays[0][img]
            get_name = get_name.replace('input\\' + dataset + '_images\\', '')
            get_num = get_name.split('_')[1]
            #if patient is same skip saving the unique position
            if unique_patient[-1] == get_num:
                continue
            unique_patient.append(get_num)
            positions.append(img)

    ratio = round(len(unique_patient) * (1 - test_size))

    # select patient 0 to ratio(th) patient
    train_img_paths = arrays[0][:positions[ratio]]
    train_mask_paths = arrays[1][:positions[ratio]]

    # select rest
    val_img_paths = arrays[0][positions[ratio]:]
    val_mask_paths = arrays[1][positions[ratio]:]

    #since we are only dealing with 110 images read them into ram right away
    train_img = []
    train_mask = []
    val_img = []
    val_mask = []

    for num in range(len(train_img_paths)):
        train_img.append(imread(train_img_paths[num]))
        train_mask.append(imread(train_mask_paths[num]))


    for num in range(len(val_img_paths)):
        val_img.append(imread(val_img_paths[num]))
        val_mask.append(imread(val_mask_paths[num]))

    split_train_img = []
    split_train_mask = []
    split_val_img = []
    split_val_mask = []
    if split>0:

        for img in train_img:
            for item in np.array_split(img, split, axis=1):
                split_train_img.append(item)

        for mask in train_mask:
            for item in np.array_split(mask, split, axis=1):
                split_train_mask.append(item)

        for img in val_img:
            for item in np.array_split(img, split, axis=1):
                split_val_img.append(item)

        for mask in val_mask:
            for item in np.array_split(mask, split, axis=1):
                split_val_mask.append(item)

    i=0
    for i in range(len(split_train_img)):
        if split_train_img[i].shape[1] < max_size:
            dif = max_size - split_train_img[i].shape[1]
            if dif > 0:
                split_train_img[i] = np.pad(split_train_img[i], [(0, 0), (0, dif)], mode='constant', constant_values=255)
                split_train_mask[i] = np.pad(split_train_mask[i], [(0, 0), (0, dif)], mode='constant', constant_values=1)

        elif split_train_img[i].shape[1] == max_size:
            continue

        else:
            print(i)
            break

    for i in range(len(split_val_img)):
        if split_val_img[i].shape[1] < max_size:
            dif = max_size - split_val_img[i].shape[1]
            if dif > 0:
                split_val_img[i] = np.pad(split_val_img[i], [(0, 0), (0, dif)], mode='constant', constant_values=255)
                split_val_mask[i] = np.pad(split_val_mask[i], [(0, 0), (0, dif)], mode='constant', constant_values=1)

        elif split_train_img[i].shape[1] == max_size:
            continue

        else:
            print(i)
            break

    random.Random(random_state).shuffle(split_train_img)
    random.Random(random_state).shuffle(split_train_mask)

    return split_train_img, split_val_img, split_train_mask, split_val_mask

def split_image (image, col_len):

    width = image.shape[1]
    num_parts = round(width/col_len)


# consider loading in all images and save them in ram then read from ram
class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, img_paths, mask_paths, crop, aug=False, ):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.crop = crop

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        #img_path = self.img_paths[idx]
        #mask_path = self.mask_paths[idx]
        image = self.img_paths[idx]
        mask = self.mask_paths[idx]
        #image = mread(img_path)
        #mask = imread(mask_path)

        #mask = cv2.normalize(mask, 1, 10, norm_type=cv2.NORM_MINMAX)

        #crop the image cutting top and bot up a bit
        if self.crop[0] < 0:
            image = image[self.crop[0]:(image.shape[0]-self.crop[1]), :]

        #space for future augmentation whatever is below is useless in this case
        if self.aug:
            if random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                image = image[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

        image = image[:, :, np.newaxis]
        image = image.transpose((2, 0, 1))
        # mask = np.squeeze(mask)
        mask = mask[:, :, np.newaxis]
        mask = mask.transpose((2, 0, 1))

        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)

        return image, mask

