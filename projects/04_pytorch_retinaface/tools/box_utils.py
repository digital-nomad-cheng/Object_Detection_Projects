import torch
import numpy as np


def xywh2ltrb(boxes):
    """Convert prior_boxes from (cx, cy, w, h) to (left, top, right, bottom)
    representation which is the format used by ground truth data.
    Args:
        boxes: (tensor) Center-size default boxes from PriorBox layers.
    Return:
        boxes: (tensor) Converted (left, top, right, bottom) form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # left, top
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # right, bottom


def ltrb2xywh(boxes):
    """ Convert prior_boxes from (left, top, right, bottom) to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) (left, top, right, bottom) format boxes
    Return:
        boxes: (tensor) Converted (cx, cy, w, h) format boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2])/2, # cx, cy
                     boxes[:, 2:] - boxes[:, :2]), 1)  # w, h


def intersect(box_a, box_b):
    """We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def IoU(box_a, box_b):
    """Compute the intersection over union between two sets of boxes.
    Here we operate on ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4]
        box_b: (tensor) Prior boxes from PriorBox layers, Shape: [num_priors, 4]
    Return:
        IoU: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def matrix_iof(a, b):
    """Return intersection over first of a and b
    numpy version for data augmentation.
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match_atss(cfg, targets, anchor_boxes, variances):
    """Match each prior box with the ground truth box of using ATSS strategy,
        encode the bounding boxes, then return the matched indices
        corresponding to both confidence and location preds.
        Args:
            targets: (list tensors) targets annotation. Shape: [batch_size, num_gts, 15]
                where index 0:4 is box annotation, index 4:14 is landmarks annotation, index
                14 is class annotation.
            anchor_boxes: (tensor) Prior boxes from priorbox layers, Shape: [n_priors, 4].
            variances: (tensor) Variances corresponding to each prior coord,
                Shape: [num_priors, 4].
        Return:
            The matched labels, boxes and landmarks for each image.
            batch_matched_labels: shape [batch_size, num_anchors], 0 represents background, 1 represents
            batch_matched_boxes: shape [batch_size, num_anchors, 4]
            batch_matched_landmarks: shape [batch_size, num_anchors, 10]
    """
    batch_size = len(targets)
    num_anchors = anchor_boxes.shape[0]

    batch_matched_labels = torch.LongTensor(batch_size, num_anchors)
    batch_matched_boxes = torch.Tensor(batch_size, num_anchors, 4)
    batch_matched_landmarks = torch.Tensor(batch_size, num_anchors, 10)
    from math import ceil
    image_size = cfg.DATA.image_size
    num_levels = len(cfg.MODEL.strides)
    feature_map_size = [[ceil(image_size[0]/step), ceil(image_size[1]/step)] for step in cfg.MODEL.strides]
    anchor_centers = anchor_boxes[:, :2].reshape(anchor_boxes.shape[0], 1, 2)
    anchors_cx_per_im = anchor_boxes[:, 0]
    anchors_cy_per_im = anchor_boxes[:, 1]


    # calculate the center distance between all anchors and ground truth boxes
    for batch_idx in range(len(targets)):
        gt_boxes = targets[batch_idx][:, :4]
        ious = IoU(
            xywh2ltrb(anchor_boxes),
            gt_boxes
        )
        cls_per_img = targets[batch_idx][:, -1]
        ldmks_per_img = targets[batch_idx][:, 4:14]
        bboxes_per_img = targets[batch_idx][:, :4]
        centers_per_img = ltrb2xywh(bboxes_per_img)[:, :2]
        target_centers = centers_per_img.reshape((1, bboxes_per_img.shape[0], 2))
        distances = (anchor_centers - target_centers).pow(2).sum(-1).sqrt()

        candidate_idxs = []
        start_idx = 0
        for i, feature_size in enumerate(feature_map_size):
            num_anchors_in_level = len(cfg.MODEL.anchor_sizes[i]) * feature_size[0] * feature_size[1]
            end_idx = start_idx + num_anchors_in_level
            distances_per_level = distances[start_idx: end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(9, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx

        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
        num_gts = bboxes_per_img.shape[0]
        candidate_ious = ious[candidate_idxs, torch.arange(num_gts)]
        iou_mean_per_gt = candidate_ious.mean(0)
        iou_std_per_gt = candidate_ious.std(0)
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
        is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

        # Limiting the final positive samples’ center to object
        for ng in range(num_gts):
            candidate_idxs[:, ng] += ng * num_anchors
        e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gts, num_anchors).contiguous().view(-1)
        e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gts, num_anchors).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)
        l = e_anchors_cx[candidate_idxs].view(-1, num_gts) - bboxes_per_img[:, 0]
        t = e_anchors_cy[candidate_idxs].view(-1, num_gts) - bboxes_per_img[:, 1]
        r = bboxes_per_img[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gts)
        b = bboxes_per_img[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gts)
        is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
        INF = 100000000
        ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        ious_inf[index] = ious.t().contiguous().view(-1)[index]
        ious_inf = ious_inf.view(num_gts, -1).t()

        anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
        matched_cls_per_img = cls_per_img[anchors_to_gt_indexs]
        matched_cls_per_img[anchors_to_gt_values == -INF] = 0
        matched_bboxes_per_img = bboxes_per_img[anchors_to_gt_indexs]
        matched_ldmks_per_img = ldmks_per_img[anchors_to_gt_indexs]

        encoded_boxes = encode(matched_bboxes_per_img, anchor_boxes, variances)
        encoded_landmarks = encode_landmark(matched_ldmks_per_img, anchor_boxes, variances)

        batch_matched_labels[batch_idx] = matched_cls_per_img  # [num_priors] top class label for each prior
        batch_matched_boxes[batch_idx] = encoded_boxes  # [num_priors, 4] encoded offsets
        batch_matched_landmarks[batch_idx] = encoded_landmarks  # [num_priors, 10] encoded landmarks

    batch_matched_labels = batch_matched_labels.cuda()
    batch_matched_boxes = batch_matched_boxes.cuda()
    batch_matched_landmarks = batch_matched_landmarks.cuda()

    return batch_matched_labels, batch_matched_boxes, batch_matched_landmarks


def match(thresholds, targets, anchor_boxes, variances):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        thresholds: (float) The overlap threshold used when mathing boxes.
        targets: (list tensors) targets annotation. Shape: [batch_size, num_gts, 15]
            where index 0:4 is box annotation, index 4:14 is landmarks annotation, index
            14 is class annotation.
        anchor_boxes: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
    Return:
        The matched labels, boxes and landmarks for each image.
        batch_matched_labels: shape [batch_size, num_anchors], 0 represents background, 1 represents
        batch_matched_boxes: shape [batch_size, num_anchors, 4]
        batch_matched_landmarks: shape [batch_size, num_anchors, 10]
    """
    batch_size = len(targets)
    num_anchors = anchor_boxes.shape[0]
    batch_matched_labels = torch.LongTensor(batch_size, num_anchors)
    batch_matched_boxes = torch.Tensor(batch_size, num_anchors, 4)
    batch_matched_landmarks = torch.Tensor(batch_size, num_anchors, 10)
    for idx in range(batch_size):
        gt_boxes = targets[idx][:, :4]
        gt_landmarks = targets[idx][:, 4:14]
        gt_labels = targets[idx][:, -1]
        overlaps = IoU(
            gt_boxes,
            xywh2ltrb(anchor_boxes)
        )

        # match each ground truth with the best IoU prior box, shape [num_objects, 1]
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

        # ignore hard gt, if the best IoU is less than 0.2, ignore this prior box
        # valid_gt_idx = best_prior_overlap[:, 0] >= 0.2  # Todo: why?

        # Todo: Test new IoU strategy
        valid_gt_idx = best_prior_overlap[:, 0] > thresholds[1]

        best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
        if best_prior_idx_filter.shape[0] <= 0:
            batch_matched_labels[idx] = 0
            batch_matched_boxes[idx] = 0
            batch_matched_landmarks[idx] = 0
            continue

        # match best ground truth for each prior box, shape [1, num_priors]
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_prior_idx.squeeze_(1)
        best_prior_idx_filter.squeeze_(1)
        best_prior_overlap.squeeze_(1)

        best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # fill with 2 to ensure best prior ?
        # ensure each gt matches with its max IoU prior box
        for j in range(best_prior_idx.size(0)):  # match prior box with ground truth box
            best_truth_idx[best_prior_idx[j]] = j

        # Todo: add positive threshold
        matched_boxes = gt_boxes[best_truth_idx]  # shape: [num_priors, 4]
        matched_labels = gt_labels[best_truth_idx]  # shape: [num_priors]
        matched_labels[best_truth_overlap < thresholds[0]] = 0  # label as background overlap < 0.35

        '''
        # Todo: Test new IoU strategy
        matched_labels[matched_labels == 1] = 0
        matched_labels[best_truth_overlap > thresholds[1]] = 1
        '''

        encoded_boxes = encode(matched_boxes, anchor_boxes, variances)
        matched_landmarks = gt_landmarks[best_truth_idx]
        encoded_landmarks = encode_landmark(matched_landmarks, anchor_boxes, variances)

        batch_matched_labels[idx] = matched_labels  # [num_priors] top class label for each prior
        batch_matched_boxes[idx] = encoded_boxes  # [num_priors, 4] encoded offsets
        batch_matched_landmarks[idx] = encoded_landmarks  # [num_priors, 10] encoded landmarks

    batch_matched_labels = batch_matched_labels.cuda()
    batch_matched_boxes = batch_matched_boxes.cuda()
    batch_matched_landmarks = batch_matched_landmarks.cuda()

    return batch_matched_labels, batch_matched_boxes, batch_matched_landmarks


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes. Refer to
    SSD paper.
    Args:
        matched: (tensor) Matched coords of ground truth for each prior in (t,l,b,r) format
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # xy distance between match center and prior's center normalized by prior's wh
    g_cxcy = ((matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]) / priors[:, 2:]
    # encode variance
    g_cxcy /= variances[0]
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    # encode variance
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors, 4]


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions from loc layers,
            Shape: [num_priors, 4]
        priors (tensor): Prior boxes in (cx, cy, w, h) form.
            Shape: [num_priors, 4].
        variances: (list[float]) Variances of prior boxes
    Return:
        decoded bounding box predictions with t,l,b,r format
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def encode_landmark(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Matched coords of ground truth for each prior in point-form
            Shape: [num_priors, 10].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded landmark (tensor), Shape: [num_priors, 10]
    """

    # x, y distance between matched landmark point and the center of prior box
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    # encode target landmark normalized by prior box w, h
    g_cxcy = (matched[:, :, :2] - priors[:, :, :2]) / priors[:, :, 2:]
    # encode variance
    g_cxcy /= variances[0]
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    # return target for smooth_l1_loss
    return g_cxcy


def decode_landmark(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of prior boxes
    Return:
        decoded landmark predictions
    """
    landmarks = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landmarks


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def py_cpu_nms(dets, threshold):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep
