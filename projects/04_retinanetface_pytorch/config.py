# config file for MobileNet0.25 and ResNet50

cfg_mnet = {
    'backbone': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrained': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channels_list': [64, 128, 256],
    'out_channels': 64
}

cfg_re50 = {
    'backbone': 'resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrained': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channels': 256,
    'out_channels': 256
}

