from easydict import EasyDict

config = EasyDict()



config.DATA = EasyDict()
config.DATA.wider_root = '/home/idealabs/data/opensource_dataset/WIDER'
config.DATA.wider_anno_path = config.DATA.wider_root + '/annotations/'
config.DATA.mtcnn_root = '/dev/data/images'
config.DATA.mtcnn_anno_path = '/dev/data/annotations'
config.DATA.smallest_face_size = 40  # the longest w/h of face used for training
# thresholds for generate neg, part and pos training data
# if IoU is below thresholds[0], it's negative
# if IoU is between thresholds[0], thresholds[1] it's part
# if IoU is above thresholds[2], it's positive
config.DATA.iou_thresholds = [0.3, 0.4, 0.65]



config.DATA.PNET = EasyDict()



config.MODEL = EasyDict()
config.MODEL.pnet_size = 12
config.MODEL.rnet_size = 24
config.MODEL.onet_size = 48
config.MODEL.stride = 2  # stride for sliding window
config.MODEL.scale_factor = 0.709  # scale factor for build image pyramids
config.MODEL.min_face_size = 12  # minimal face size to detect


config.TRAIN = EasyDict()
config.TRAIN.use_gpu = True
config.TRAIN.gpu_device = 1
config.TRAIN.manual_seed = 2020
config.TRAIN.batch_size = 10240
config.TRAIN.optimizer = "SGD"
config.TRAIN.epochs = 50
config.TRAIN.LR = EasyDict()
config.TRAIN.LR.initial_lr = 1e-2
config.TRAIN.LR.momentum = 0.9
config.TRAIN.LR.gamma = 0.1
config.TRAIN.LR.weight_decay = 5e-4
config.TRAIN.LR.decay_epochs = [5, 30, 40, 50]
config.TRAIN.use_landmarks = False


config.TRAIN.pnet_pretrained = 'best_pnet.pth'


# test parameters
config.TEST = EasyDict()
config.TEST.use_gpu = True
config.TEST.gpu_device = 1
config.TEST.pnet_model = './pretrained_weights/best_pnet.pth'
config.TEST.rnet_model = './pretrained_weights/best_rnet.pth'
config.TEST.onet_model = './pretrained_weights/best_onet.pth'
config.TEST.nms_thresholds = [0.7, 0.8, 0.9]
config.TEST.score_thresholds = [0.7, 0.8, 0.9]
config.TEST.min_face_size = 12

