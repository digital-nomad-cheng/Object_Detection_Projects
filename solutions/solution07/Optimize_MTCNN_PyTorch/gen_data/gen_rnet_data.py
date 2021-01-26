"""
Generate positive, negative, positive images whose size are rnet_size*rnet_size from PNet.
"""
import os, sys
sys.path.append('.')

import cv2
import numpy as np
import torch

from nets.mtcnn import MTCNNDetector
from tools.utils import*
from configs.mtcnn_config import config


def generate_rnet_data(cfg: object, mode: object = 'train') -> object:
    net_size = cfg.MODEL.rnet_size
    prefix = ''
    anno_file = os.path.join(cfg.DATA.wider_anno_path, "wider_anno_{}.txt".format(mode))
    im_dir = os.path.join(cfg.DATA.wider_root, "WIDER_{}/images".format(mode))
    pos_save_dir = os.path.join(cfg.DATA.mtcnn_root, "{}/{}/positive".format(mode, net_size))
    part_save_dir = os.path.join(cfg.DATA.mtcnn_root, "{}/{}/part".format(mode, net_size))
    neg_save_dir = os.path.join(cfg.DATA.mtcnn_root, "{}/{}/negative".format(mode, net_size))

    if not os.path.exists(pos_save_dir):
        os.makedirs(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.makedirs(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.makedirs(neg_save_dir)

    # store labels of positive, negative, part images
    if not os.path.exists(cfg.DATA.mtcnn_anno_path):
        os.makedirs(cfg.DATA.mtcnn_anno_path)
    f1 = open(os.path.join(cfg.DATA.mtcnn_anno_path, 'pos_{}_{}.txt'.format(net_size, mode)), 'w')
    f2 = open(os.path.join(cfg.DATA.mtcnn_anno_path, 'neg_{}_{}.txt'.format(net_size, mode)), 'w')
    f3 = open(os.path.join(cfg.DATA.mtcnn_anno_path, 'part_{}_{}.txt'.format(net_size, mode)), 'w')

    # anno_file: store labels of the wider face training data
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print("%d pics in total" % num)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # part
    idx = 0

    # create MTCNN Detector, only use pnet for detection
    mtcnn_detector = MTCNNDetector(cfg)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_path = os.path.join(prefix, annotation[0])
        print(im_path)
        bbox = list(map(float, annotation[1:]))
        boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
        # anno form is x1, y1, w, h, convert to x1, y1, x2, y2
        boxes[:,2] += boxes[:,0] - 1
        boxes[:,3] += boxes[:,1] - 1

        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = mtcnn_detector.detect_face(image)

        if bboxes.shape[0] == 0:
            continue

        dets = np.round(bboxes[:, 0:4])

        img = cv2.imread(im_path)
        idx += 1

        height, width, channel = img.shape

        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, boxes)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (net_size, net_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.2 and n_idx < 3.0*p_idx+1:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx_Iou = np.argmax(Iou)
                assigned_gt = boxes[idx_Iou]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4 and d_idx < 1.0*p_idx + 1:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1

        print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

        # if idx == 20:
        #     break

    f1.close()
    f2.close()
    f3.close()


if __name__ == "__main__":
    # only use pnet for detection
    config.TEST.rnet_model = None
    config.TEST.onet_model = None
    config.TEST.thresholds = [0.6, 0.7, 0.7]
    # generate_rnet_data(config, 'train')
    # generate_rnet_data(config, 'val')