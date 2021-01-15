from easydict import EasyDict


config = EasyDict()
config.TRAIN = EasyDict()
config.TRAIN.num_gpus = 2
config.TRAIN.use_gpu = True
config.TRAIN.batch_size = 64
config.TRAIN.epochs = 250
# use pretrained backbone network
config.TRAIN.pretrained = True
# box loss weight
config.TRAIN.box_loss_weight = 2.0
# box encoding variance for x, y and w, h
config.TRAIN.encode_variance = [0.1, 0.2]
# clip box after encoding
config.TRAIN.clip_box = True

config.MODEL = EasyDict()
config.MODEL.backbone = "mobilenet0.25"
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