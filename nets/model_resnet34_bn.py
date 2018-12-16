import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from torch import optim
from utils.loss import FocalLoss2d, RobustFocalLoss2d
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from utils.metrics import accuracy_and_dice
from utils.loss_factory import get_loss

#  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#  resnet18 :  BasicBlock, [2, 2, 2, 2]
#  resnet34 :  BasicBlock, [3, 4, 6, 3]
#  resnet50 :  Bottleneck  [3, 4, 6, 3]
#

# https://medium.com/neuromation-io-blog/deepglobe-challenge-three-papers-from-neuromation-accepted-fe09a1a7fa53
# https://github.com/ternaus/TernausNetV2
# https://github.com/neptune-ml/open-solution-salt-detection
# https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution


##############################################################3
#  https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py
#  https://pytorch.org/docs/stable/torchvision/models.html

import torchvision


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.bn = SynchronizedBatchNorm2d(out_channels)


    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x





class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x ):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        return x

#
# resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
# resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
# resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
# resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
# resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'

class UNetResNet34(nn.Module):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.


    def load_pretrain(self, pretrain_file):
        self.resnet.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self, num_classes, criterion='BCE', activation='sigmoid', class_weights=None):
        super().__init__()
        self.resnet = torchvision.models.resnet34()
        self.num_classes = num_classes
        self.criterion_type = criterion

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )# 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  #128
        self.encoder4 = self.resnet.layer3  #256
        self.encoder5 = self.resnet.layer4  #512

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(512+256, 512, 256)
        self.decoder4 = Decoder(256+256, 512, 256)
        self.decoder3 = Decoder(128+256, 256,  64)
        self.decoder2 = Decoder( 64+ 64, 128, 128)
        self.decoder1 = Decoder(128    , 128,  32)

        self.logit    = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, padding=0),
        )
        loss_parameters = {
            'weight': class_weights,
            # 'activation': activation,
        }
        self.criterion_loss = get_loss(criterion)(**loss_parameters)
        self.activation = activation

    def forward(self, x):
        #batch_size,C,H,W = x.shape

        # mean=[0.485, 0.456, 0.406]
        # std =[0.229, 0.224, 0.225]
        # x = torch.cat([
        #     (x-mean[0])/std[0],
        #     (x-mean[1])/std[1],
        #     (x-mean[2])/std[2],
        # ],1)


        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        e2 = self.encoder2( x)  #; print('e2',e2.size())
        e3 = self.encoder3(e2)  #; print('e3',e3.size())
        e4 = self.encoder4(e3)  #; print('e4',e4.size())
        e5 = self.encoder5(e4)  #; print('e5',e5.size())


        #f = F.max_pool2d(e5, kernel_size=2, stride=2 )  #; print(f.size())
        #f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)#False
        #f = self.center(f)                       #; print('center',f.size())
        f = self.center(e5)
         
        f = self.decoder5(torch.cat([f, e5], 1))  #; print('d5',f.size())
        f = self.decoder4(torch.cat([f, e4], 1))  #; print('d4',f.size())
        f = self.decoder3(torch.cat([f, e3], 1))  #; print('d3',f.size())
        f = self.decoder2(torch.cat([f, e2], 1))  #; print('d2',f.size())
        f = self.decoder1(f)                      # ; print('d1',f.size())

        #f = F.dropout2d(f, p=0.20)
        logit = self.logit(f)                     #; print('logit',logit.size())
        return logit


    ##-----------------------------------------------------------------


    def criterion(self, logit, truth):

        #loss = PseudoBCELoss2d()(logit, truth)
        #loss = FocalLoss2d()(logit, truth, type='sigmoid')
        # loss = RobustFocalLoss2d()(logit, truth, type='sigmoid')
        if 'cross' not in self.criterion_type:
            truth = truth.unsqueeze(1).float()
            truth = truth.float()
            loss = self.criterion_loss(logit, truth)
        else:
            truth = truth.view(-1).long()
            logit = logit.permute((0, 2, 3, 1)).contiguous().view(-1, self.num_classes)
            loss = self.criterion_loss(logit, truth)
        return loss


    # def criterion(self,logit, truth):
    #
    #     loss = F.binary_cross_entropy_with_logits(logit, truth)
    #     return loss



    def metric(self, logit, truth, num_categories, threshold=0.5, neg_thresh=0):
        if self.num_classes == 1:
            prob = F.sigmoid(logit)
            prob = prob.squeeze(1)
            result = accuracy_and_dice(prob > threshold, truth, num_categories, neg_thresh)
        else:
            prob = F.softmax(logit, dim=1)
            pred = prob.max(1)[1].squeeze(1)
            # dice, acc, recall = accuracy_and_dice(pred, truth, num_categories)
            result = accuracy_and_dice(pred, truth, num_categories, neg_thresh)
        return result



    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


def unet34(num_classes, criterion='BCE', activation='sigmoid', init_function=None, class_weights=None):
    model = UNetResNet34(num_classes, criterion, activation, class_weights)
    if init_function is None:
        print('='*20)
        print('Warning! The model did not initializes!')
    else:
        model.apply(init_function)
    return model

# SaltNet = UNetResNet34



### run ##############################################################################



def run_check_net():

    batch_size = 8
    C,H,W = 1, 128, 128

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (2,   (batch_size,C,H,W)).astype(np.float32)


    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).float().cuda()


    #---
    net = SaltNet().cuda()
    net.set_mode('train')
    # print(net)
    # exit(0)

    #net.load_pretrain('/root/share/project/kaggle/tgs/data/model/resnet50-19c8e357.pth')

    logit = net(input)
    loss  = net.criterion(logit, truth)
    dice  = net.metric(logit, truth)

    print('loss : %0.8f'%loss.item())
    print('dice : %0.8f'%dice.item())
    print('')


    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(net.parameters(), lr=0.001)


    i=0
    optimizer.zero_grad()
    while i<=500:

        logit = net(input)
        loss  = net.criterion(logit, truth)
        dice  = net.metric(logit, truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%20==0:
            print('[%05d] loss, dice  :  %0.5f,%0.5f'%(i, loss.item(),dice.item()))
        i = i+1







########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print( 'sucessful!')