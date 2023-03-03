import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.resnet

import torch.nn.init as init


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


PAD = "<PAD>"  # padding
EOS = "<EOS>"  # end of sequence
SOS = "<SOS>"  # start of sequence

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2

torch.manual_seed(1)
CUDA = torch.cuda.is_available()


def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x


def mask_triu(x):  # mask out subsequent positions
    y = Tensor(np.triu(np.ones([x.size(2), x.size(2)]), 1)).byte()
    return torch.gt(x + y, 0)


def mask_pad(x):  # mask out padded positions
    return x.data.eq(PAD_IDX).view(x.size(0), 1, 1, -1)


class Pos_Encoder(nn.Module):
    def __init__(self, Embedding_Size, maxlen=4096):
        super(Pos_Encoder, self).__init__()
        pe = Tensor(maxlen, Embedding_Size)
        pos = torch.arange(0, maxlen, 1.).unsqueeze(1)
        k = torch.exp(np.log(10000) * -torch.arange(0,
                      Embedding_Size, 2.) / Embedding_Size)
        pe[:, 0::2] = torch.sin(pos * k)
        pe[:, 1::2] = torch.cos(pos * k)
        pe = nn.Parameter(pe)
        self.register_parameter('pe', pe)

    def forward(self, n):
        return self.pe[:n]


def attention_sdp(Dim_K, q, k, v, mask):
    c = np.sqrt(Dim_K)
    attention_wight = torch.matmul(q, k.transpose(2, 3)) / c
    if mask is not None:
        mask = mask.unsqueeze(1)
        attention_wight = attention_wight.masked_fill(mask == 0, -1e9)
    attention_wight = F.softmax(attention_wight, 3)
    a = torch.matmul(attention_wight, v)

    return a, attention_wight  # attention weights


class Attention_Blosk(nn.Module):
    def __init__(self, Dim_K, Dim_V, Embedding_Size=512, Num_Head=8, Dropout=0.1):
        super(Attention_Blosk, self).__init__()
        self.Dim_K = Dim_K
        self.Dim_V = Dim_V
        self.Num_Head = Num_Head
        self.Wq = nn.Linear(Embedding_Size, Dim_K * Num_Head)
        self.Wk = nn.Linear(Embedding_Size, Dim_K * Num_Head)
        self.Wv = nn.Linear(Embedding_Size, Dim_V * Num_Head)

        self.Wout = nn.Linear(Dim_V * Num_Head, Embedding_Size)
        self.dropout = nn.Dropout(Dropout)
        self.LayerNorm = nn.LayerNorm(Embedding_Size)

    def forward(self, q, k, v, mask):
        x = q
        batch_size = q.shape[0]
        q = self.Wq(q).view(batch_size, -1, self.Num_Head,
                            self.Dim_K).transpose(1, 2)
        k = self.Wk(k).view(batch_size, -1, self.Num_Head,
                            self.Dim_K).transpose(1, 2)
        v = self.Wv(v).view(batch_size, -1, self.Num_Head,
                            self.Dim_V).transpose(1, 2)
        attention_out, attention_wight = attention_sdp(
            self.Dim_K, q, k, v, mask)

        attention_out = attention_out.transpose(1, 2).contiguous().view(
            batch_size, -1, self.Num_Head * self.Dim_V)

        attention_out = self.Wout(attention_out)

        attention_out = self.dropout(attention_out)

        attention_out = self.LayerNorm(x + attention_out)
        return attention_out, attention_wight


class Feed_Forward_Net(nn.Module):
    def __init__(self, Embedding_Size, Mid_Size, Dropout=0.1):
        super(Feed_Forward_Net, self).__init__()
        self.Linear1 = nn.Linear(Embedding_Size, Mid_Size)
        self.RelU = nn.ReLU()
        self.Linear2 = nn.Linear(Mid_Size, Embedding_Size)

        self.dropout = nn.Dropout(Dropout)
        self.LayerNorm = nn.LayerNorm(Embedding_Size)

    def forward(self, x):
        res = x
        output = self.Linear1(x)
        output = self.RelU(output)
        output = self.Linear2(output)
        output = self.dropout(output)
        output = self.LayerNorm(res + output)
        return output


class Encoder_Layer(nn.Module):
    def __init__(self, Dim_K, Dim_V, Embedding_Size=512, Num_Head=8):
        super(Encoder_Layer, self).__init__()
        self.attention = Attention_Blosk(
            Dim_K, Dim_V, Embedding_Size=Embedding_Size, Num_Head=Num_Head)
        self.Feed_Forward_Net = Feed_Forward_Net(Embedding_Size, Mid_Size=1024)

    def forward(self, x, mask):
        output, attention_ = self.attention(x, x, x, mask)
        output = self.Feed_Forward_Net(output)
        return output, attention_


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_layers, Dim_K, Dim_V, Embedding_Size=512, Num_Head=8):
        super(Encoder, self).__init__()

        self.Embedding = nn.Linear(vocab_size, Embedding_Size)
        self.Pos_Encoding = Pos_Encoder(Embedding_Size)
        self.Encoder_Layers = nn.ModuleList([Encoder_Layer(Dim_K, Dim_V,
                                                           Embedding_Size=Embedding_Size,
                                                           Num_Head=Num_Head) for _ in range(num_layers)])

    def forward(self, x, mask):
        output = self.Embedding(x)
        pos = self.Pos_Encoding(output.size(1))
        output = output + pos
        for Layer in self.Encoder_Layers:
            output, attention_ = Layer(output, mask)
        return output


class Decoder_Layer(nn.Module):
    def __init__(self, Dim_K, Dim_V, Embedding_Size=512, Num_Head=8):
        super(Decoder_Layer, self).__init__()

        self.attention1 = Attention_Blosk(
            Dim_K, Dim_V, Embedding_Size=Embedding_Size, Num_Head=Num_Head)
        self.attention2 = Attention_Blosk(
            Dim_K, Dim_V, Embedding_Size=Embedding_Size, Num_Head=Num_Head)
        self.Feed_Forward_Net = Feed_Forward_Net(Embedding_Size, 1024)

    def forward(self, Encoder_out, Decoder_in, mask1, mask2=None):
        output, _ = self.attention1(Decoder_in, Decoder_in, Decoder_in, mask1)
        output, attention_wight = self.attention2(
            output, Encoder_out, Encoder_out, mask2)

        output = self.Feed_Forward_Net(output)
        return output, attention_wight


class Decoder(nn.Module):
    def __init__(self, vocab_size, num_layers, Dim_K, Dim_V, Embedding_Size=512, Num_Head=8):
        super(Decoder, self).__init__()
        self.Embedding = nn.Embedding(vocab_size, Embedding_Size,
                                      padding_idx=PAD_IDX)

        self.Pos_Encoding = Pos_Encoder(Embedding_Size)
        self.Decoder_Layers = nn.ModuleList([Decoder_Layer(Dim_K, Dim_V,
                                                           Embedding_Size=Embedding_Size,
                                                           Num_Head=Num_Head) for _ in range(num_layers)])

    def forward(self, Encoder_out, Decoder_in, mask1, mask2=None):
        output = self.Embedding(Decoder_in)
        output = output + self.Pos_Encoding(output.size(1))
        for Layer in self.Decoder_Layers:
            output, attention_wight = Layer(Encoder_out, output, mask1, mask2)
        return output, attention_wight


class Generator(nn.Module):
    def __init__(self, vocab_size, Embedding_Size=512):
        super(Generator, self).__init__()
        self.out = nn.Linear(Embedding_Size, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.out(x), dim=-1)


def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.upsample(x, size=(H, W), mode='bilinear') + y


def _upsample(x, y):
    _, _, H, W = y.size()
    return F.upsample(x, size=(H, W), mode='bilinear')


class Fpn(nn.Module):
    def __init__(self):
        super(Fpn, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            2048, 256, kernel_size=1, stride=1, padding=0)
        self.bn_layer1 = nn.BatchNorm2d(256)
        self.relu_layer1 = nn.ReLU(inplace=True)

        self.conv_layer2 = nn.Conv2d(
            1024, 256, kernel_size=1, stride=1, padding=0)
        self.bn_layer2 = nn.BatchNorm2d(256)
        self.relu_layer2 = nn.ReLU(inplace=True)

        self.conv_layer3 = nn.Conv2d(
            512, 256, kernel_size=1, stride=1, padding=0)
        self.bn_layer3 = nn.BatchNorm2d(256)
        self.relu_layer3 = nn.ReLU(inplace=True)

        self.conv_layer4 = nn.Conv2d(
            256, 256, kernel_size=1, stride=1, padding=0)
        self.bn_layer4 = nn.BatchNorm2d(256)
        self.relu_layer4 = nn.ReLU(inplace=True)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_bn = nn.BatchNorm2d(256)
        self.smooth1_relu = nn.ReLU(inplace=True)

        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_bn = nn.BatchNorm2d(256)
        self.smooth2_relu = nn.ReLU(inplace=True)

        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_bn = nn.BatchNorm2d(256)
        self.smooth3_relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv2_pos = nn.Conv2d(2, 512, 1, padding=0)
        self.bn2_pos = nn.BatchNorm2d(512)

    def forward(self, f):
        layer1_out = self.conv_layer1(f[3])
        layer1_out = self.bn_layer1(layer1_out)
        layer1_out = self.relu_layer1(layer1_out)

        layer2_out = self.conv_layer2(f[2])
        layer2_out = self.bn_layer2(layer2_out)
        layer2_out = self.relu_layer2(layer2_out)

        layer3_out = self.conv_layer3(f[1])
        layer3_out = self.bn_layer3(layer3_out)
        layer3_out = self.relu_layer3(layer3_out)

        layer4_out = self.conv_layer4(f[0])
        layer4_out = self.bn_layer4(layer4_out)
        layer4_out = self.relu_layer4(layer4_out)

        p0 = layer1_out

        p1 = _upsample_add(layer1_out, layer2_out)
        p1 = self.smooth1(p1)
        p1 = self.smooth1_bn(p1)
        p1 = self.smooth1_relu(p1)

        p2 = _upsample_add(p1, layer3_out)
        p2 = self.smooth2(p2)
        p2 = self.smooth2_bn(p2)
        p2 = self.smooth2_relu(p2)

        p3 = _upsample_add(p2, layer4_out)
        p3 = self.smooth3(p3)
        p3 = self.smooth3_bn(p3)
        p3 = self.smooth3_relu(p3)

        p0 = _upsample(p0, p3)
        p1 = _upsample(p1, p3)
        p2 = _upsample(p2, p3)

        h = torch.cat((p0, p1, p2, p3), 1)

        h = self.conv1(h)
        h = self.relu1(self.bn1(h))

        h = self.maxpool(h)

        h_return = h

        feature_w, feature_h = h.size(2), h.size(3)
        pos_x = list(range(0, feature_w))
        pos_x = np.asarray(pos_x) / feature_w
        pos_x = pos_x.reshape(len(pos_x), 1)

        pos_y = list(range(0, feature_h))
        pos_y = np.asarray(pos_y) / feature_h
        pos_y = pos_y.reshape(len(pos_y), 1)

        ones = np.ones([h.size(0), 1, feature_w, feature_h])
        pos_x = pos_x * ones
        pos_y = np.transpose(pos_y) * ones

        pos_x = torch.Tensor(pos_x)
        pos_y = torch.Tensor(pos_y)

        pos_xy = torch.cat([pos_x, pos_y], 1)
        if CUDA:
            pos_xy = pos_xy.cuda()

        pos_xy = self.conv2_pos(pos_xy)
        pos_xy = self.bn2_pos(pos_xy)
        pos_xy = F.sigmoid(pos_xy)
        h = h + pos_xy

        input_encoder = h.view(h.size(0), h.size(1), -1).permute(0, 2, 1)
        return input_encoder, h_return


class Res_Encoder(nn.Module):
    def __init__(self):
        super(Res_Encoder, self).__init__()
        self.resnet = model.resnet.resnet50(True)
        self.fpn = Fpn()
        self.fpn.apply(weights_init)

    def forward(self, x):
        _, f = self.resnet(x)
        input_encoder, h_return = self.fpn(f)
        return input_encoder, h_return
