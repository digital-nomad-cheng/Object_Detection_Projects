from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg: EasyDict):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = cfg.MODEL.num_classes
        self.threshold = cfg.TRAIN.overlap_thresholds
        self.do_neg_mining = cfg.TRAIN.do_negative_mining
        self.neg_pos_ratio = cfg.TRAIN.neg_pos_ratio
        self.variance = cfg.TRAIN.encode_variance
        self.use_gpu = cfg.TRAIN.use_gpu

    def forward(self, predictions, prior_boxes, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and landmark preds from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            prior_boxes (tensor): torch.size(num_priors, 4), prior box generated
                from PriorBox
            targets (tensor): Ground truth boxes, landmarks and label for a batch,
                shape: [batch_size,num_objs, 4+1+2*5] (last idx is the label).
        """

        pred_logits, pred_boxes, pred_landmarks = predictions
        batch_size = pred_boxes.size(0)
        num_priors = (prior_boxes.size(0))

        # match prior boxes with ground truth boxes
        loc_t = torch.Tensor(batch_size, num_priors, 4)
        landm_t = torch.Tensor(batch_size, num_priors, 10)
        conf_t = torch.LongTensor(batch_size, num_priors)
        for idx in range(batch_size):
            gt_boxes = targets[idx][:, :4].data
            gt_landmarks = targets[idx][:, 4:14].data
            gt_labels = targets[idx][:, -1].data
            defaults = prior_boxes.data  # Todo: why data
            # match prior box with predictions
            match(self.threshold, gt_boxes, defaults, self.variance, gt_labels, gt_landmarks, loc_t, conf_t, landm_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        # 1. calculate landmarks loss for anchors which are above IoU threshold with ground truth boxes
        zeros = torch.tensor(0).cuda()
        pos_ldmks_mask = conf_t > zeros  # anchors with IoU below threshold are set label 0
        num_pos_ldmks = pos_ldmks_mask.long().sum(1, keepdim=True)
        N1 = max(num_pos_ldmks.data.sum().float(), 1)
        pos_ldmks_idx = pos_ldmks_mask.unsqueeze(pos_ldmks_mask.dim()).expand_as(pred_landmarks)
        pos_ldmks_pred = pred_landmarks[pos_ldmks_idx].view(-1, 10)
        pos_ldmks_gt = landm_t[pos_ldmks_idx].view(-1, 10)
        # landmark smooth l1 loss, shape: [batch_size, num_prior_boxes, 10]
        landmark_loss = F.smooth_l1_loss(pos_ldmks_pred, pos_ldmks_gt, reduction='sum')

        # 2. calculate box loss for anchors which are above IoU threshold with ground truth boxes
        pos_boxes_mask = conf_t != zeros  # Todo: are there any that is below zero?
        conf_t[pos_boxes_mask] = 1
        pos_boxes_idx = pos_boxes_mask.unsqueeze(pos_boxes_mask.dim()).expand_as(pred_boxes)
        pos_boxes_pred = pred_boxes[pos_boxes_idx].view(-1, 4)
        pos_boxes_gt = loc_t[pos_boxes_idx].view(-1, 4)
        # Localization smooth l1 loss, shape: [batch_size, num_prior_boxes, 4]
        box_loss = F.smooth_l1_loss(pos_boxes_pred, pos_boxes_gt, reduction='sum')

        # 3. Online Hard Negative Mining, keep balance of nums of positive and negative samples
        # Compute max conf across batch for hard negative mining
        batch_conf = pred_logits.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c[pos_boxes_mask.view(-1, 1)] = 0  # filter out positive boxes and only keep negative boxes
        loss_c = loss_c.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos_boxes_mask.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio*num_pos, max=pos_boxes_mask.size(1)-1)
        neg_boxes_mask = idx_rank < num_neg.expand_as(idx_rank)

        # 4. Classification cross entropy loss with balanced positive and negative loss
        pos_boxes_idx = pos_boxes_mask.unsqueeze(2).expand_as(pred_logits)
        neg_boxes_idx = neg_boxes_mask.unsqueeze(2).expand_as(pred_logits)
        cls_pred = pred_logits[(pos_boxes_idx+neg_boxes_idx).gt(0)].view(-1, self.num_classes)
        cls_gt = conf_t[(pos_boxes_mask+neg_boxes_mask).gt(0)]
        cls_loss = F.cross_entropy(cls_pred, cls_gt, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        box_loss /= N
        cls_loss /= N
        landmark_loss /= N1

        return cls_loss, box_loss, landmark_loss
