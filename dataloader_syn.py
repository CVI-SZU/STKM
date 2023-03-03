import random

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import os
import sys
import numpy as np
import cv2
from torchvision import transforms

import alphabet
str1 = alphabet.alphabet


def str_Converter_init():
    dict = {"PAD": 0, "SOS": 1, "EOS": 2, "Blank": 3}
    for i, char in enumerate(str1):
        # NOTE: 0 is reserved for 'blank' required by wrap_ctc
        dict[char] = i + 4
    nclass = len(str1) + 5
    return dict, nclass


def str_Converter(label, dict):
    if dict.__contains__(label):
        return dict[label]
    else:
        return len(str1) + 4


def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img


def rotate_img(img, angle_range=10):
    center_x = (img.shape[1] - 1) // 2
    center_y = (img.shape[0] - 1) // 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)  # 12
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img


def subsequent_mask(size):
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad=0):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return Variable(tgt_mask, requires_grad=False)


def mask_gen(size, lable_lenght):
    mask = np.triu(np.ones((size, size)), k=1).astype('uint8')
    total_lenght = 1
    lable_lenght_next = 0
    for i in range(len(lable_lenght) - 1):
        lable_lenght_next = lable_lenght[i+1]
        total_lenght += (lable_lenght[i] + 1)
        mask[total_lenght:total_lenght+lable_lenght[i+1], :total_lenght] = 1
    mask[total_lenght+lable_lenght_next+1:, :total_lenght] = 1
    mask = torch.from_numpy(mask) == 0
    return mask


def resize_padding(image, w=512):
    max_wh = max(image.shape[0], image.shape[1])
    newImage = np.zeros((max_wh, max_wh, 3), np.uint8)
    newImage[:image.shape[0], :image.shape[1], :] = image
    newImage = cv2.resize(newImage, (w, w))
    return newImage


def extract_vertices(lines, dict):
    lenght_lable = []
    labels = []
    labels_ones = []
    for line in lines[1:]:
        label = line.rstrip('\n').lstrip('\ufeff')
        labels.append(label)

    labels = sorted(labels)

    for label_ in labels:
        if label_ != "###":
            for i in range(len(label_)):
                labels_ones.append(str_Converter(label_[i], dict))
            labels_ones.append(2)
            lenght_lable.append(len(label_))
    return labels_ones, lenght_lable


class synthtext_dataset(data.Dataset):
    def __init__(self, img_path, gt_path, len_img=512, batch_max_line=16, batch_max_length=400):
        super(synthtext_dataset, self).__init__()
        self.img_path = img_path
        self.gt_files = [os.path.join(gt_path, gt_file)
                         for gt_file in sorted(os.listdir(gt_path))]
        self.len_img = len_img
        self.batch_max_line = batch_max_line
        self.batch_max_length = batch_max_length
        self.dict, self.nclass = str_Converter_init()

        print(len(self.gt_files))

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        gt_path = self.gt_files[index]
        with open(gt_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        img = cv2.imread(
            str(self.img_path + lines[0].rstrip('\n').lstrip('\ufeff')), cv2.IMREAD_COLOR)
        img = resize_padding(img)
        if random.random() < 0.5:
            rotate_angle = random.randint(-10, 10)
            img = rotate_img(img, rotate_angle)
        img = img / 255.0
        img = torch.Tensor(img).permute(2, 0, 1)

        tags, _ = extract_vertices(lines, self.dict)
        mask = subsequent_mask(self.batch_max_length)

        tags_y = tags[:]
        tags_y.append(3)
        label = np.zeros(self.batch_max_length, dtype=int)
        label[0] = 1
        for i in range(len(tags)):
            label[i + 1] = tags[i]
        label = torch.from_numpy(label)

        label_y = np.zeros(self.batch_max_length, dtype=int)
        for i in range(len(tags_y)):
            label_y[i] = tags_y[i]
        label_y = torch.from_numpy(label_y)

        tgt_mask = (label != 0).unsqueeze(-2)
        mask = tgt_mask & Variable(mask.type_as(tgt_mask.data))

        return transform(img), label, label_y, Variable(mask, requires_grad=False)


class test_load(data.Dataset):
    def __init__(self, img_path, len_img=512):
        self.img_files = [os.path.join(img_path, img_file)
                          for img_file in sorted(os.listdir(img_path))]
        self.len_img = len_img

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img = cv2.imread(self.img_files[index], cv2.IMREAD_COLOR)
        img = resize_padding(img)
        img = img/255.0

        img = torch.Tensor(img).permute(2, 0, 1)
        return transform(img)
