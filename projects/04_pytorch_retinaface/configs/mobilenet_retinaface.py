from easydict import EasyDict

config = EasyDict()
config.TRAIN = EasyDict()
config.TRAIN.num_gpus = 2
config.TRAIN.use_gpu = True
config.TRAIN.batch_size = 64
config.TRAIN.num_workers = 12
config.TRAIN.epochs = 250
# use pretrained backbone network
config.TRAIN.pretrained = True
# box loss weight
config.TRAIN.box_loss_weight = 2.0
# box encoding variance for x, y and w, h
config.TRAIN.encode_variance = [0.1, 0.2]
# clip box after encoding
config.TRAIN.clip_box = True
config.TRAIN.overlap_thresholds = [0.35, 0.4]
# whether perform negative sample mining
config.TRAIN.do_negative_mining = True
# number of negative sample vs positive samples
config.TRAIN.neg_pos_ratio = 3
# warmup epochs
config.TRAIN.warmup_epochs = 10

config.TRAIN.LR = EasyDict()
config.TRAIN.LR.initial_lr = 1e-3
config.TRAIN.LR.momentum = 0.9
config.TRAIN.LR.gamma = 0.1
config.TRAIN.LR.weight_decay = 5e-4
config.TRAIN.LR.decay_epoch1 = 190
config.TRAIN.LR.decay_epoch2 = 220

config.MODEL = EasyDict()
config.MODEL.backbone = "mobilenet0.25"
config.MODEL.num_classes = 2
# anchor size in the original image
config.MODEL.anchor_sizes = [[16, 32], [64, 128], [256, 512]]
# anchor feature map strides
config.MODEL.strides = [8, 16, 32]
# feature map input channels
config.MODEL.in_channels_list = [64, 128, 256]
# output channels in FPN and SSH
config.MODEL.out_channels = 64
config.MODEL.return_layers = {'stage1': 1, 'stage2': 2, 'stage3': 3}

config.DATA = EasyDict()
# training image size, if the image size is not fixed, we have to generate anchors for each input
config.DATA.image_size = 640
config.DATA.rgb_mean = (127.5, 127.5, 127.5)


cfg_mnet = {
    'backbone': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 64,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrained': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channels_list': [64, 128, 256],
    'out_channels': 64
}