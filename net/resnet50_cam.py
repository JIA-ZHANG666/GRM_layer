import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
import math
import os
from torch.nn import Parameter
from step.sgr_layer import * 

import pickle
import json

from graph.grm_layer import * 
from graph import *
from graph.voc_data import *


class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, input):
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                            training=False, eps=self.eps)

class Net(nn.Module):

    def __init__(self, stride=16, n_classes=20, t=0.4, in_channel=300,dropout = 0.5 , input_feature_channels=256,  visual_feature_channels =256,
                 adj_file='./data/voc/voc_adj.pkl'):
        super(Net, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        self.dropout = dropout

        ####################
        voc_data = get_voc_data()
        self.gc1 = GRMLayer(2048, 2048, **voc_data)
        self.gc_add = nn.ModuleList([self.gc1])
        ##################################
    
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])
        


    #def forward(self, x):
    def forward(self, x, inp):

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        
        x = self.stage4(x)
        x = torchutils.gap2d(x, keepdims=True)#torch.Size([8, 2048, 1, 1])
       
        xf = x.view(x.size(0),-1)#fecture_Net: torch.Size([8, 2048])
        
        inp = inp[0]
       
        G = self.gc1(inp, self.S)#G_Net1111: torch.Size([20, 1024])
        G = self.relu(G)
       
        G = self.gc2(G, self.S)#G_Net2222: torch.Size([20, 2048])
        G = self.relu(G)
        G = G.transpose(0, 1)#G_Net: torch.Size([2048, 20])
        #################
        
        x1 = torch.matmul(xf, G)
        
        x = self.classifier(x)
        #print("---------x_classifier_out:",x1.size())
        x = x.view(-1, self.n_classes)

        return x,x1
        #return x

    def train(self, mode=True):
        super(Net, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        """return (list(self.backbone.parameters()), list(self.newly_added.parameters()), list(self.newly_Conv.parameters()),
        list(self.newly_gc1.parameters()),list(self.newly_gc2.parameters()),list(self.newly_gc3.parameters()) )"""
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()), 
        list(self.newly_gc1.parameters()), list(self.newly_gc2.parameters()))

class Net_CAM(Net):

    def __init__(self,stride=16,n_classes=20):
        super(Net_CAM, self).__init__(stride=stride,n_classes=n_classes)
        
    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)

        x = torchutils.gap2d(feature, keepdims=True)
        
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)
        
        return x,cams,feature

class Net_CAM_Feature(Net):

    def __init__(self,stride=16,n_classes=20):
        super(Net_CAM_Feature, self).__init__(stride=stride,n_classes=n_classes)
        
    #def forward(self, x):
    def forward(self, x, inp):

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        
        feature = self.stage4(x) # bs*2048*32*32
        
        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
       
        x = x.view(-1, self.n_classes)

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)
        cams = cams/(F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
        cams_feature = cams.unsqueeze(2)*feature.unsqueeze(1) # bs*20*2048*32*32
        cams_feature = cams_feature.view(cams_feature.size(0),cams_feature.size(1),cams_feature.size(2),-1)
        cams_feature = torch.mean(cams_feature,-1)
        
        return x,cams_feature,cams
        

    """def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]"""

class CAM(Net):

    def __init__(self, stride=16,n_classes=20):
        super(CAM, self).__init__(stride=stride,n_classes=n_classes)
        #self.Conv = nn.Conv2d(2048, 256, 1, bias=False)

    def forward(self, x, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, self.classifier.weight)
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    def forward1(self, x, weight, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, weight)
        
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    def forward2(self, x, weight, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, weight*self.classifier.weight)
        
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)
        return x

class Class_Predictor(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(Class_Predictor, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(representation_size, num_classes, 1, bias=False)

    def forward(self, x, label):
        batch_size = x.shape[0]
        x = x.reshape(batch_size,self.num_classes,-1) # bs*20*2048
        mask = label>0 # bs*20

        feature_list = [x[i][mask[i]] for i in range(batch_size)] # bs*n*2048
        prediction = [self.classifier(y.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for y in feature_list]
        labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]

        loss = 0
        acc = 0
        num = 0
        for logit,label in zip(prediction, labels):
            if label.shape[0] == 0:
                continue
            loss_ce= F.cross_entropy(logit, label)
            loss += loss_ce
            acc += (logit.argmax(dim=1)==label.view(-1)).sum().float()
            num += label.size(0)
            
        return loss/batch_size,acc/num
        #return loss/batch_size, acc/(num+1e-9)
