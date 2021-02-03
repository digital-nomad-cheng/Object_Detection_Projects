from easydict import EasyDict

config = EasyDict()
config.TRAIN = EasyDict()
config.TRAIN.num_gpus = 1
config.TRAIN.use_gpu = True
config.TRAIN.batch_size = 32
config.TRAIN.num_workers = 12
config.TRAIN.epochs = 250
# use pretrained backbone network
config.TRAIN.pretrained = True
# box loss weight
config.TRAIN.box_loss_weight = 2.0
# box encoding variance for x, y and w, h
config.TRAIN.encode_variance = [0.1, 0.2]
# clip box after encoding
config.TRAIN.clip_box = False
config.TRAIN.overlap_thresholds = [0.35, 0.5]
# choose classification loss between Online Hard Example Mining or Focal Loss
config.TRAIN.cls_loss_type = "OHEM"  # "OHEM" or "FocalLoss"
# ratio of negative sample vs positive samples if choose OHEM
config.TRAIN.neg_pos_ratio = 7
# warmup epochs
config.TRAIN.warmup_epochs = 10
# learning rate related parameters
config.TRAIN.LR = EasyDict()
config.TRAIN.LR.initial_lr = 1e-3
config.TRAIN.LR.momentum = 0.9
config.TRAIN.LR.gamma = 0.1
config.TRAIN.LR.weight_decay = 5e-4
config.TRAIN.LR.decay_epoch1 = 190
config.TRAIN.LR.decay_epoch2 = 220

config.MODEL = EasyDict()
# config.MODEL.backbone = "mobilenet0.25"
config.MODEL.num_classes = 2
# anchor size in the original image
config.MODEL.anchor_sizes = [[32, 64, 128], [256], [512]]
# anchor feature map strides
config.MODEL.strides = [32, 64, 128]

config.DATA = EasyDict()
# training image size, if the image size is not fixed, we have to generate anchors for each input
config.DATA.image_size = (1024, 1024)
config.DATA.rgb_mean = (127.5, 127.5, 127.5)

config.TEST = EasyDict()
config.TEST.confidence_threshold = 0.02
config.TEST.nms_threshold = 0.4
config.TEST.top_k = 5000
config.TEST.keep_top_k = 750

