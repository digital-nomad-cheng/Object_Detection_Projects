1. 通过对比基于锚框的多阶段法Faster R-CNN和基于锚框的单阶段法SSD，列举出多阶段法

   Faster R-CNN性能好于单阶段法SSD的几点原因。

   + 单阶段SSD主要的问题之一是训练的时候类别不平衡的问题，而多阶段Faster RCNN RPN 网络对后面的Fast RCNN网络来说相当于对样本采样过程，可以过滤掉大部分的负样本，使得 Faster RCNN 的后半部分没有样本类别不平衡的问题能训练出更好的效果。

   + 单阶段SSD只针对anchor box进行了一次偏移量的回归，而Faster RCNN先经过 RPN 回归一次，后面再回归一次，相当于经过了两次回归，可以得到更准确的物体位置。

   + 多阶段 Faster RCNN 相对于单阶段 SSD 来说多了两个 loss 函数，一个函数是针对 background 和 non-background 分类，另一个是针对 anchor 进行回归，有点类似于多任务网络，多一个分支来对网络进行监督从而训练出更 generalized 的 feature map。

     

2. 通过对比基于锚框的单阶段法RetinaNet和无需锚框的中心域法FCOS，列举出两者之间的不同

   之处。

   + feature map 每个位置对应的 anchor 数量，RetinaNet是每个位置铺设多个不同尺度和不同大小的 anchor box，而 FCOS  feature map 每个位置只有一个 anchor point。
   + 正负样本的定义。RetinaNet 通过 ground truth 和 anchor box 之间的 IoU 来定义正负样本，如阈值高于一定值为正，低于一定值为负。而 FCOS 使用空间和尺度限制来定义正负样本，在 ground truth 的 bounding box 内的 anchor point 作为 candidate, 再使用不同feature map检测不同scale物体来筛选限制。
   + 回归的起始状态，RetinaNet 从铺设的 anchor box 的四个顶点开始回归到真实标注的四个顶点的偏移量，FCOS 从 anchor point 开始回归到真实标注四条边的距离。

