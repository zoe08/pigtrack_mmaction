import logging
import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)),
                              requires_grad=True)  # nn.Parameter将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.func = nn.Softmax()  # 二分类


    def forward(self, x, s=64, m=0.5):  # s=64, m=222.5为超参数m为弧度
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)  # 传入的参数nn.Parameter在0维上进行标准化

        cosa = torch.matmul(x_norm, w_norm) / s  # torch.matmul二维的点成，高维的矩阵乘法
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(s * torch.cos(a + m))\
                     / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(a + m)))  # 代码实现公式
        out = torch.log(arcsoftmax)
        return out  # arcsoftmax,

class Net(nn.Module):
    def __init__(self, num_classes = 88, reid=True):
        super(Net, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 64 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 64 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 128 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 256 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # 512 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 512 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes)  # num_classes)
        )
        # 新加 为faceloss
        self.arc_softmax = Arcsoftmax(512, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        feature = self.layer2(x)
        x = self.layer3(feature)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        embedding = x
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        else:
            # # classifier
            x = self.classifier(x)
            # x = self.arc_softmax(x)
            return embedding, x

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=torch.device(self.device))[
            'net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    net = Net()
    x = torch.randn(2, 3, 128, 64)
    _, y = net(x)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir='scalar2')

    dummy_input = torch.rand(1, 3, 128, 64)
    input = torch.autograd.Variable(dummy_input)

    writer.add_graph(net, input)

    writer.close()
    # import ipdb
    # ipdb.set_trace()
    # 用于调试，运行则能跳到ipython里 (Ipython Debugger)


