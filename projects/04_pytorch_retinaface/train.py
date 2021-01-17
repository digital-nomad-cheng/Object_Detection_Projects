from __future__ import print_function
import os
import time
import datetime
import math
import argparse

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from tools.dataset import WiderFaceDetection, detection_collate
from tools.data_augment import preproc
from configs.mobilenet_retinaface import config as cfg
from layers.loss import MultiBoxLoss
from layers.prior_box import PriorBox
from nets.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--dataset_root', default='./data/widerface/train/', help='Training dataset directory')
parser.add_argument('--anno_file', default='label.txt', help='Training dataset annotation file')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

training_dataset = args.dataset_root
anno_file = args.anno_file
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if cfg.TRAIN.num_gpus > 1 and cfg.TRAIN.use_gpu:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True

optimizer = optim.SGD(
    net.parameters(), lr=cfg.TRAIN.LR.initial_lr, momentum=cfg.TRAIN.LR.momentum, weight_decay=cfg.TRAIN.LR.weight_decay)
criterion = MultiBoxLoss(cfg)

prior_box = PriorBox(cfg)
with torch.no_grad():
    priors = prior_box.forward()
    priors = priors.cuda()


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset, anno_file, preproc(cfg.DATA.image_size, cfg.DATA.rgb_mean))

    epoch_size = math.ceil(len(dataset) / cfg.TRAIN.batch_size)
    max_iter = cfg.TRAIN.epochs * epoch_size

    decay_steps = (cfg.TRAIN.LR.decay_epoch1 * epoch_size, cfg.TRAIN.LR.decay_epoch2 * epoch_size)
    gamma_step = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(
                data.DataLoader(
                    dataset, cfg.TRAIN.batch_size, shuffle=True, num_workers=cfg.TRAIN.num_workers, collate_fn=detection_collate
                )
            )

            torch.save(net.state_dict(), save_folder + cfg.MODEL.backbone + '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        if iteration in decay_steps:
            gamma_step += 1
        lr = adjust_learning_rate(cfg, optimizer, epoch, gamma_step, iteration, epoch_size)

        loader_t0 = time.time()
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        out = net(images)
        optimizer.zero_grad()
        cls_loss, box_loss, landmark_loss = criterion(out, priors, targets)
        loss = cls_loss + cfg.TRAIN.box_loss_weight*box_loss + landmark_loss
        loss.backward()
        optimizer.step()
        loader_t1 = time.time()

        batch_time = loader_t1 - loader_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{:03d}/{} || Iter: {:05d}/{} || Cls: {:8.4f} Box: {:8.4f} Landmark: {:8.4f} || LR: {:.8f} || \
            Batchtime: {:.4f} s || ETA: {}'.format(epoch, cfg.TRAIN.epochs, iteration + 1, max_iter,
                                                   cls_loss.item(), box_loss.item(), landmark_loss.item(),
                                                   lr, batch_time, str(datetime.timedelta(seconds=eta))
                )
        )

    torch.save(net.state_dict(), os.path.join(save_folder, cfg.MODEL.backbone + '_final.pth'))


def adjust_learning_rate(cfg, optimizer, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epochs = cfg.TRAIN.warmup_epochs
    initial_lr = cfg.TRAIN.LR.initial_lr
    gamma = cfg.TRAIN.LR.gamma
    if epoch <= warmup_epochs:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epochs)
    else:
        lr = initial_lr * (gamma ** step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
