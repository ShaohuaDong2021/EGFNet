import os
import shutil
import json
import time

from apex import amp
import apex
import copy

import numpy as np
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader

from toolbox import MscCrossEntropyLoss
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt
from toolbox import Ranger
from toolbox import setup_seed
from toolbox import load_ckpt
from toolbox import group_weight_decay
from toolbox import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, ProbOhemCrossEntropy2d, FocalLoss2d, \
    LovaszSoftmax, LDAMLoss
setup_seed(33)


class eeemodelLoss(nn.Module):

    def __init__(self, class_weight=None, ignore_index=-100, reduction='mean'):
        super(eeemodelLoss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
            [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()
        self.class_weight_binary = torch.from_numpy(np.array([1.5121, 10.2388])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4459, 23.7228])).float()

        self.class_weight = class_weight
        self.LovaszSoftmax = LovaszSoftmax()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        # self.semantic_loss = nn.CrossEntropyLoss()
        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)


    def forward(self, inputs, targets):
        # semantic_out, binary_out, boundary_out = inputs
        semantic_gt, binary_gt, boundary_gt = targets
        semantic_out, binary_out, boundary_out = inputs
        loss1 = self.semantic_loss(semantic_out, semantic_gt)
        loss2 = self.binary_loss(binary_out, binary_gt)
        loss3 = self.boundary_loss(boundary_out, boundary_gt)
        loss = loss1 + loss2 + loss3
        return loss



def run(args):
    torch.cuda.set_device(args.cuda)
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'/mnt/Data1/shaohuadong/model/vggmask/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    # model
    model = get_model(cfg)
    device = torch.device(f'cuda:{args.cuda}')
    model.to(device)

    # dataloader
    trainset, _, testset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)
    # val_loader = DataLoader(valset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
    #                         pin_memory=True)
    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                             pin_memory=True)

    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    # criterion = LovaszSoftmax().to(device)
    train_criterion = eeemodelLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # 指标 包含unlabel
    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    best_test = 0

    amp.register_float_function(torch, 'sigmoid')
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # 每个epoch迭代循环
    for ep in range(cfg['epochs']):

        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            if cfg['inputs'] == 'rgb':
                image = sample['image'].to(device)
                label = sample['label'].to(device)
                bound = sample['bound'].to(device)
                edge = sample['edge'].to(device)
                binary_label = sample['binary_label'].to(device)
                targets = [label, binary_label, bound]
                predict = model(image)
            else:
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)
                bound = sample['bound'].to(device)
                edge = sample['edge'].to(device)
                binary_label = sample['binary_label'].to(device)
                targets = [label, binary_label, bound]
                predict = model(image, depth, edge)
                # predict = model(image, depth)
            loss = train_criterion(predict, targets)
            ####################################################

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

        scheduler.step(ep)

        # test
        with torch.no_grad():
            model.eval()
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader):
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    edge = sample['edge'].to(device)
                    predict = model(image)
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)
                    edge = sample['edge'].to(device)
                    # predict = model(image, depth)[0]
                    predict = model(image, depth)
                    # predict = model(image, depth, edge)[0]
                    # predict = model(image, depth)

                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())

                predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)

        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg

        test_macc = running_metrics_test.get_scores()[0]["class_acc: "]
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]
        test_avg = (test_macc + test_miou) / 2

        logger.info(
            f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] loss={train_loss:.3f}/{test_loss:.3f}, mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}')
        if test_avg > best_test:
            best_test = test_avg
            save_ckpt(logdir, model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="configs/EGFNet.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbd'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--cuda", type=int, default=0, help="set cuda device id")

    args = parser.parse_args()

    run(args)

