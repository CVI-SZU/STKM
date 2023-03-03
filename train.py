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
from model.transformers import Res_Encoder, Decoder, Generator
from dataloader_syn import *
from loss import *
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0,
                    help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=10,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100,
                    help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=12,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--len_img", type=int, default=512,
                    help="size of image height")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument("--bn_dropout_off_interval", type=int,
                    default=500, help="interval between saving generator outputs")
parser.add_argument("--sample_interval", type=int, default=500,
                    help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=3000,
                    help="interval between saving model checkpoints")
opt = parser.parse_args()
print(opt)
dict, num_class = str_Converter_init()
print(num_class)
str1 = alphabet.alphabet
print(str1)
print(len(str1))


def draw_features(attention_weight, savename, img_self):
    img_self = img_self[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
    img_self = ((img_self * 0.5) + 0.5) * 255
    img_self = img_self.astype(np.uint8)
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
            './test_results/img/atten_map_{}_{}.jpg'.format(savename, i), img_out)


def eval_img(batches_done, start_symbol=1):
    imgs = next(iter(test_loader))
    imgs = imgs.cuda()
    res_encoder.eval()
    decoder.eval()
    generator.eval()
    encoder_output, _ = res_encoder(imgs)
    encoder_feature = encoder_output.contiguous().permute(0, 2, 1).view(
        encoder_output.size(0), encoder_output.size(2), 64, 64)

    encoder_feature = torch.mean(encoder_feature, dim=1)
    word_last = 0
    attention_wight_all = []
    attention_wight_all.append(encoder_feature)
    # attention_wight_all.append(torch.mean(h, dim=1))
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
    with open('./test_results/txt/answer_{}.txt'.format(batches_done), 'w') as f:
        f.write(result)
    f.close()
    draw_features(attention_wight_all, batches_done, imgs)


if __name__ == '__main__':
    manualSeed = random.randint(1, 30000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    res_encoder_path = None
    decoder_path = None
    generator_path = None

    train_img_path = "data/SynthText/img/"
    train_gt_path = "data/SynthText/gt/"

    test_img_path = os.path.abspath("data/test/img/")
    test_gt_path = os.path.abspath("data/test/gt/")

    file_train_num = len(os.listdir(train_gt_path))

    trainset = synthtext_dataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(trainset,
                                   batch_size=opt.batch_size,
                                   shuffle=True,
                                   num_workers=opt.n_cpu,
                                   pin_memory=True)

    testset = test_load(test_img_path)
    test_loader = data.DataLoader(testset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=1,
                                  pin_memory=True)

    cuda = torch.cuda.is_available()
    criterion = LabelSmoothing(size=num_class, padding_idx=0, smoothing=0.1)
    res_encoder = Res_Encoder()
    decoder = Decoder(vocab_size=num_class, num_layers=1, Dim_K=32, Dim_V=32)
    generator = Generator(vocab_size=num_class)
    data_parallel = False

    if cuda:
        res_encoder = res_encoder.cuda()
        decoder = decoder.cuda()
        generator = generator.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        if res_encoder_path is not None:
            print("loading res_encoder model")
            res_encoder.load_state_dict(torch.load(res_encoder_path))
            print("OK!")

        if decoder_path is not None:
            print("loading decoder model")
            decoder.load_state_dict(torch.load(decoder_path))
            print("OK!")

        if generator_path is not None:
            print("loading generator model")
            generator.load_state_dict(torch.load(generator_path))
            print("OK!")

        if torch.cuda.device_count() > 1:
            res_encoder = nn.DataParallel(res_encoder)
            decoder = nn.DataParallel(decoder)
            generator = nn.DataParallel(generator)
            data_parallel = True

    optimizer = torch.optim.Adam(itertools.chain(res_encoder.parameters(),
                                                 decoder.parameters(),
                                                 generator.parameters()),
                                 lr=opt.lr,
                                 betas=(opt.b1, opt.b2))

    prev_time = time.time()

    for epoch in range(opt.epoch, opt.n_epochs):
        total_tokens = 0
        total_loss = 0
        tokens = 0
        train_iter = iter(train_loader)
        i = 0
        res_encoder.train()
        decoder.train()
        generator.train()
        while i < len(train_loader):

            i += 1
            imgs, label, label_y, trg_mask = train_iter.next()
            imgs, label, label_y, trg_mask = imgs.cuda(
            ), label.cuda(), label_y.cuda(), trg_mask.cuda()
            ntokens = (label_y != 0).data.sum()
            encoder_output, _ = res_encoder(imgs)

            output, _ = decoder(encoder_output, label, trg_mask)
            output = generator(output)
            loss = criterion(output.contiguous().view(-1, output.size(-1)),
                             label_y.contiguous().view(-1)) / ntokens

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data * ntokens
            total_tokens += ntokens
            tokens += ntokens

            batches_done = epoch * (len(train_loader) - 7) + i
            batches_left = opt.n_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time) / batches_done)

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [total loss: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(train_loader),
                    loss.item(),
                    total_loss / total_tokens,
                    time_left,
                )
            )

            if batches_done % opt.bn_dropout_off_interval == 0:
                for module in [res_encoder, decoder, generator]:
                    for m in module.modules():
                        if isinstance(m, (nn.BatchNorm2d, nn.Dropout)):
                            m.eval()

            if opt.checkpoint_interval != -1 and batches_done % opt.checkpoint_interval == 0:
                # Save model
                res_encoder_state_dict = res_encoder.module.state_dict(
                ) if data_parallel else res_encoder.state_dict()
                torch.save(res_encoder_state_dict,
                           "./saved_models/res_encoder_%d.pth" % batches_done)
                decoder_state_dict = decoder.module.state_dict(
                ) if data_parallel else decoder.state_dict()
                torch.save(decoder_state_dict,
                           "./saved_models/decoder_%d.pth" % batches_done)
                generator_state_dict = generator.module.state_dict(
                ) if data_parallel else generator.state_dict()
                torch.save(generator_state_dict,
                           "./saved_models/generator_%d.pth" % batches_done)
