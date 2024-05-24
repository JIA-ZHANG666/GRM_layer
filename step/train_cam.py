import cv2

import torch
import torch.nn as nn
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils
from torch import autograd
import os
import keras.backend as K
import math
import numpy as np
Tensor = torch.Tensor()


def FocalLoss(y_pred, y_true, gamma=2):
    # y_pred is the logits before Sigmoid
    assert y_pred.shape == y_true.shape
    pt = torch.exp(-F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')).detach()
    sample_weight = (1 - pt) ** gamma
    return F.binary_cross_entropy_with_logits(y_pred, y_true, weight=sample_weight)

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()
    ce = nn.CrossEntropyLoss()
    criterion = FocalLoss()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)
            inp_var = pack['inp']

            x,x1 = model(img, inp_var)
            
            loss = F.multilabel_soft_margin_loss(x, label)
            loss2 = criterion(x1, label)
            loss = loss + 0.0001*loss1
           


            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')()


    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root, inp_name=args.inp_name,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root, inp_name=args.inp_name,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},

        #{'params': param_groups[2], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[2], 'lr': 0.1*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[3], 'lr': 0.1*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        #{'params': param_groups[5], 'lr': 0.1*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()
    ce = nn.CrossEntropyLoss()
    criterion = FocalLoss()
  
    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            #img = torch.autograd.Variable(pack['img']).float().cuda()
            img = img.cuda()
            label = pack['label'].cuda(non_blocking=True)
            #label = torch.autograd.Variable(pack['label']).float().cuda()
            inp_var = pack['inp'].cuda()

            x,x1 = model(img, inp_var)

            optimizer.zero_grad()

            loss = F.multilabel_soft_margin_loss(x, label)
            loss1 = criterion(x1, label)
            loss = loss + 0.0001*loss1

            loss.backward()
            avg_meter.add({'loss': loss.item()})

            nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
            optimizer.step()
            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      #'lr_G: %.4f' % (optimizer.param_groups[2]['lr'])
                      #'etc:%s' % (timer.str_estimated_complete()), flush=True
                      )

        
        validate(model, val_data_loader)
        timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name)
    torch.cuda.empty_cache()
