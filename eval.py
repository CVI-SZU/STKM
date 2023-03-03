import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import random
import alphabet
from torch.backends import cudnn
from torch.optim import lr_scheduler
from model.transformers import *
from dataset import *
from loss import *

dict, num_class = str_Converter_init()
print(num_class)
str1 = alphabet.alphabet
print(str1)
print(len(str1))


def load_img(img):
    transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img = torch.Tensor(img).permute(2, 0, 1)
    return transform(img).unsqueeze(0)


def draw_features(attention_weight, savename, img_self):
    img_self = img_self[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
    img_self = ((img_self * 0.5) + 0.5) * 255
    img_self = img_self.astype(np.uint8)
    cv2.imwrite('./eval_result/img_{}.jpg'.format(savename), img_self)
    for i in range(len(attention_weight)):
        atten_once = attention_weight[i]
        atten_once = atten_once.permute(1, 2, 0).cpu().detach().numpy()
        img = atten_once[:, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        h, w = img.shape[:2]
        img = cv2.resize(img, (512, int(h * 512 / w)))
        img1 = cv2.addWeighted(img_self, 0.5, img, 0.5, 0)
        img_out = np.hstack((img_self, img1, img))
        cv2.imwrite(
            './eval_result/atten_map_{}_{}.jpg'.format(savename, i), img_out)


def eval_img(image_path, start_symbol=1):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (512, 512))
    cv2.imwrite('./eval_result/img_{}.jpg'.format(1), img)
    img = img / 255.0
    imgs = load_img(img)

    res_encoder = Res_Encoder()
    decoder = Decoder(vocab_size=num_class, num_layers=1, Dim_K=32, Dim_V=32)
    generator = Generator(vocab_size=num_class)
    res_encoder_path = './saved_models/res_encoder_120000.pth'
    decoder_path = './saved_models/decoder_120000.pth'
    Generator_path = './saved_models/generator_120000.pth'
    cuda = torch.cuda.is_available()
    if cuda:
        res_encoder = res_encoder.cuda()
        res_encoder.load_state_dict(torch.load(res_encoder_path))
        decoder = decoder.cuda()
        decoder.load_state_dict(torch.load(decoder_path))
        generator = generator.cuda()
        generator.load_state_dict(torch.load(Generator_path))
        imgs = imgs.cuda()

    res_encoder.eval()
    decoder.eval()
    generator.eval()
    encoder_output, h = res_encoder(imgs)
    encoder_feature = encoder_output.contiguous().permute(0, 2, 1).view(
        encoder_output.size(0), encoder_output.size(2), 64, 64)

    encoder_feature = torch.mean(encoder_feature, dim=1)
    word_last = 0
    attention_wight_all = []
    attention_wight_all.append(encoder_feature)
    attention_wight_once = torch.zeros((1, 64, 64)).cuda()
    ys = torch.ones(1, 1).fill_(start_symbol).long().cuda()
    lenght_char = 0
    lenght_srt = [0]
    for i in range(100 - 1):
        out, attention_wight = decoder(encoder_output,
                                       Variable(ys),
                                       Variable(subsequent_mask(ys.size(1)).unsqueeze(0).long().cuda()))
        prob = generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.ones(1, 1).long().cuda().fill_(next_word)], dim=1)
        attention_wight = attention_wight[:, :, -1,
                                          :].view(attention_wight.size(0), -1, 64, 64)
        attention_wight = torch.mean(attention_wight, dim=1)

        if word_last == 2:
            lenght_char = 0
            lenght_srt.append(lenght_char)
            attention_wight_all.append(attention_wight_once)
            attention_wight_once = torch.zeros((1, 64, 64)).cuda()

        if next_word.item() == 3:
            break

        if next_word.item() != 2:
            lenght_char += 1
            lenght_srt.pop()
            lenght_srt.append(lenght_char)
            attention_wight_once += attention_wight

        word_last = next_word.item()
    ret = ys.cpu().numpy()[0]
    result = []

    for i in ret:
        if i >= num_class:
            result.append("*")
        if i == 2:
            result.append("\n")
        if 3 < i < num_class - 1:
            result.append(str1[i - 4])
    result = "".join(result[0:])
    with open('./eval_result/answer_{}.txt'.format(1), 'w') as f:
        f.write(result)
    f.close()
    draw_features(attention_wight_all, 1, imgs)


if __name__ == '__main__':
    image_path = "./img_test.jpg"
    eval_img(image_path)
