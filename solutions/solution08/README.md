

## 模型构建

1. 配置文件:

   ```python
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
   ```

   主要注意两点，后面的锚框密集化操作和这两个参数有关

   ```python
   config.MODEL.anchor_sizes = [[32, 64, 128], [256], [512]] # 每个层铺设的 anchor 大小，第一个层铺设了三种尺度的 anchor
   # anchor feature map strides
   config.MODEL.strides = [32, 64, 128] # 每个特征层对应的 stride
   
   
   ```

2. 网络：

   ![image-20210203201613835](/Users/vincent/Library/Application Support/typora-user-images/image-20210203201613835.png)

   Faceboxes 的网络主要由两部分组成

   + Rapidly Digested Convolutional Layers: 使用大的卷积kernel_size，大的 stride 来快速减小输入的尺寸。大kernel_size要配合大stride同时使用，这样不会因为stride过大使得空间像素没有充分利用。同时利用浅层激活层油很强的负相关性使用CReLU来降低卷机channel数量。
   + Multiple Scale Convolutional Layers: 使用Inception模块对特征进行增强，同时在多个尺度检测。

3. Faceboxes 第二个不同的地方在于锚框的生成，有一个锚框密集化操作。

   <img src="/Users/vincent/Library/Application Support/typora-user-images/image-20210203201646375.png" alt="image-20210203201646375" style="zoom:67%;" />

   锚框密集程度 = $anchor\_size / stride$,这个公式可以看出密度分别为1, 2, 4, 4, 4,因此对anchor_size为32, 64的框进行4倍，8倍密集化操作。

   ```python
   if min_size == 32:
     dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
     dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
     for cy, cx in product(dense_cy, dense_cx):
       anchors += [cx, cy, s_kx, s_ky]
   elif min_size == 64:
     dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
     dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
     for cy, cx in product(dense_cy, dense_cx):
       anchors += [cx, cy, s_kx, s_ky]
   ```

## 优化器构建

这里和原始Faceboxes实现不太一样，优化器是SGD，具体参数都在下面。

```python
config.TRAIN.warmup_epochs = 10
# learning rate related parameters
config.TRAIN.LR = EasyDict()
config.TRAIN.LR.initial_lr = 1e-3
config.TRAIN.LR.momentum = 0.9
config.TRAIN.LR.gamma = 0.1
config.TRAIN.LR.weight_decay = 5e-4
config.TRAIN.LR.decay_epoch1 = 190
config.TRAIN.LR.decay_epoch2 = 220
```



## 数据加载器

这部分重点部分是数据增强的部分，对代码进行解释下。





## 结论

重新实现了一遍 FaceBoxes，然后自己训练了下，差距较大，问题在哪还在排查，结果如下:

+ 和原始实现的差别主要在于直接对网络输入缩放到 [-1, 1]，而不是减去（104， 117， 123） bgr order
+ 训练的 epoch 数目更少， 只有 250 epoch

| DATASET |   AP   |
| :-----: | :----: |
|  FDDB   | 94.48% |
| PASCAL  | 96.56% |
|   AFW   | 98.92% |



*图片大小的放大对检测结果的影响很大，FDDB和PASCAL数据不进行3倍和2.5倍放大，测试结果直接拉垮。*



