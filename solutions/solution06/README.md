## 参数解析

```yaml
MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN" 
  WEIGHT: "catalog://ImageNetPretrained/MSRA/R-50" # imagenet 预训练模型
  RPN_ONLY: True # ?
  ATSS_ON: True # 使用 Adaptive Training Sample Selection 策略
  BACKBONE:
    CONV_BODY: "R-50-FPN-RETINANET" # backbone + FPN 网络结构
  RESNETS:
    BACKBONE_OUT_CHANNELS: 256
  RETINANET:
    USE_C5: False # ?
  ATSS:
    ANCHOR_SIZES: (64, 128, 256, 512, 1024) # 8S ?
    ASPECT_RATIOS: (1.0,) # ATSS 里面不用配置 anchor 的长宽比
    SCALES_PER_OCTAVE: 1 # ATSS 里面不用配置 anchor 的大小
    USE_DCN_IN_TOWER: False # 是否使用 deformable convolution 
    POSITIVE_TYPE: 'ATSS' # how to select positves: ATSS (Ours) , SSC (FCOS), IoU (RetinaNet)
    TOPK: 9 # 每一层选取的候选 anchor 数量
    REGRESSION_TYPE: 'BOX' # 回归点还是框，ATSS 只是一种 anchor 自动匹配策略，可以适应两种，BOX是RetinaNet，POINT是FCOS
DATASETS:
  TRAIN: ("coco_2017_train",)
  TEST: ("coco_2017_val",)
INPUT:
  MIN_SIZE_TRAIN: (800,)
  MAX_SIZE_TRAIN: 1333
  MIN_SIZE_TEST: 800
  MAX_SIZE_TEST: 1333
DATALOADER:
  SIZE_DIVISIBILITY: 32
SOLVER:
  BASE_LR: 0.01
  WEIGHT_DECAY: 0.0001
  STEPS: (60000, 80000)
  MAX_ITER: 90000
  IMS_PER_BATCH: 4
  WARMUP_METHOD: "constant"
```

## 运行demo

1. 安装

   ```bash
   conda create -n ATSS python=3.7
   conda activate ATSS
   conda install ipython
   pip install ninja yacs cython matplotlib tqdm
   conda install -c pytorch torchvision cudatoolkit=10.0
   conda install -c conda-forge pycocotools
   git clone https://github.com/sfzhang15/ATSS.git
   cd ATSS
   python setup.py build develop --no-deps
   ```
   ![00_install_atss](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/00_install_atss.png)

2. 运行 demo

   ```python demo/atss_demo.py```

   ![01_atss-demo](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/01_atss-demo.png)



## 调试

1. 

1. 构建模型

   ![03_build_model](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/03_build_model.png)

2. 构建 backbone 特征提取器，这里没有区分 backbone 和 neck 所以 resnet和fpn是一起构建的

   ![04_build_backbone](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/04_build_backbone.png)

3. 构建 ATSS head 在增强后的特征基础上进行分类和回归，同时增加了centerness分支。至此模型构建完成开始模型训练。

   ![05_build_atss](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/05_build_atss.png)

4. generalized_rcnn.py中的forward函数: 这里可以看出大致的运行流程，先将images转换为ImageList，经过backbone提取特征，通过rpn获取proposals和proposal_losses, 这里的rpn就是ATSSModule，因此核心逻辑都在atss.py里面。后面着重分析atss.py和loss.py两个函数。

   ![06_rcnn_forward](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/06_rcnn_forward.png)

5. atss.py 中 BoxCoder是对训练的目标进行编码，当 `cfg.MODEL.ATSS.REGRESSION_TYPE` 为 `POINT` 的时候，回归的目标是anchor中心点到真实框四条边的距离。同样预测的时候按照相反的规则进行解码

   ![07_boxcoder](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/07_boxcoder.png)



### ATSS

接下来主要看看 loss.py 关于 anchor 和 ground truth 的匹配，以及 threshold 计算等核心逻辑都在这里。

1. atss.py 中 ATSSHead 是对 feature pyramid 中的 cell 进行 classification， box_regression 和 centerness prediction

   ![08_atsshead](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/08_atsshead.png)

2. atss.py 中 ATSSModule 主要是对 anchor 进行分类和回归同时计算 loss

   ![09_atssmodule](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/09_atssmodule.png)

3. GIoU loss GIoU 是对 IoU 的改进，可以直接用 1 - $GIoU$ 作为 loss 函数来训练模型 

   $GIoU$  = $IoU - \frac{|Ac - U|}{|Ac|}$

   先计算两个框之间的最小外接矩形面积$Ac$, 计算两个框的所占的面积$U$, $Ac - U$ 是最小外接矩形不属于两个框的面积。和$IoU$相比$GIoU$是更好的距离度量指标，不仅仅关注重叠区域也关注非重叠区域。

   ![GIoU](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/GIoU.png)

4. 先针对每一层计算每个 anchor 到 ground truth 中心点距离，然后根据距离排序选出 topk 候选 anchor  ![10_selectopk](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/10_selectopk.png)

5. 计算 ground truth 和 候选 anchor. 之间的 IoU, 计算这些候选 anchor IOU 的均值和标准差，筛选该 ground truth 的候选 anchor 的 IoU 阈值时 mean + std

   ![11_adaptive_thresh](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/11_adaptive_thresh.png)

6. 使用该 IoU 阈值筛选 anchor 同时保证 anchor 中心在 ground truth 里面

1. ![12_in_center](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/12_in_center.png)

​	

## 问题

1. PyCharm debug torch.distributed.launch debug 的时候报错，可以正常运行

   ```python -m torch.distributed.launch``` 实际运行的是另外一个torch库脚本，路径不在当前工程里面。修改 script path 可以正常调试。

   ![13_debug](/Users/vincent/Documents/Repo/Object_Detection_Projects/solutions/solution06/breakpoint_debugging_images/13_debug.png)

   