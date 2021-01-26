import os, sys
sys.path.append('.')

import torch

from tools.dataset import FaceDataset
from nets.modules import RNet
from training.rnet.trainer import RNetTrainer
from configs.mtcnn_config import config as cfg

# set device
use_cuda = cfg.TRAIN.use_gpu and torch.cuda.is_available()
torch.cuda.manual_seed(cfg.TRAIN.manual_seed)
device = torch.device("cuda:{}".format(cfg.TRAIN.gpu_device) if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# build dataloader
kwargs = {'num_workers': 8, 'pin_memory': False} if use_cuda else {}
train_data = FaceDataset(os.path.join(cfg.DATA.mtcnn_anno_path, 'imglist_anno_24_train.txt'))
val_data = FaceDataset(os.path.join(cfg.DATA.mtcnn_anno_path, 'imglist_anno_24_val.txt'))
dataloaders = {'train': torch.utils.data.DataLoader(train_data,
                        batch_size=cfg.TRAIN.batch_size, shuffle=True, **kwargs),
               'val': torch.utils.data.DataLoader(val_data,
                        batch_size=cfg.TRAIN.batch_size, shuffle=False, **kwargs)
              }

# build model
model = RNet(is_train=True)
model = model.to(device)
# model.load_state_dict(torch.load('pretrained_weights/slim_mtcnn/best_pnet.pth'), strict=True)

# build checkpoint
# checkpoint = CheckPoint(train_config.save_path)

# build optimzier
if cfg.TRAIN.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR.initial_lr, momentum=cfg.TRAIN.LR.momentum, weight_decay=cfg.TRAIN.LR.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR.initial_lr, momentum=cfg.TRAIN.LR.momentum, weight_decay=cfg.TRAIN.LR.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.LR.decay_epochs, gamma=0.1)


# build trainer
trainer = RNetTrainer(cfg.TRAIN.epochs, dataloaders, model, optimizer, scheduler, device)

# train
trainer.train()
