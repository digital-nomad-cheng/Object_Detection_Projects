from __future__ import print_function
import os
import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from configs.mobilenet_retinaface import config as cfg
from layers.anchor import AnchorGenerator
from nets.retinaface import RetinaFace
from tools.box_utils import decode, decode_landmark, py_cpu_nms
from tools.timer import Timer

parser = argparse.ArgumentParser(description='RetinaFace')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--save_folder', default='./model_eval/widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_threshold', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module.' """
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')

    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    dataset_folder = args.dataset_folder
    anno_file = "label.txt"

    # read images from annotation file
    image_list = []
    with open(os.path.join(dataset_folder, anno_file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                line = line[2:]
                image_list.append(line)

    num_images = len(image_list)

    timer = {'forward_pass': Timer(), 'misc': Timer()}

    for i, img_name in enumerate(image_list):
        image_path = os.path.join(dataset_folder, "images", img_name)
        img_bgr = cv2.imread(image_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img = np.float32(img)

        im_height, im_width, _ = img.shape
        cfg.DATA.image_size = img.shape[0:2]
        img -= cfg.DATA.rgb_mean
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)

        torch.cuda.synchronize()
        timer['forward_pass'].tic()
        logits, offsets, landmarks = net(img)  # forward pass
        torch.cuda.synchronize()
        timer['forward_pass'].toc()
        timer['misc'].tic()
        anchor_generator = AnchorGenerator(cfg)
        anchors = anchor_generator.generate_anchors()
        anchors = anchors.to(device)
        boxes = decode(offsets.data.squeeze(0), anchors, cfg.TRAIN.encode_variance)
        scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]])
        boxes = boxes * scale.to(device)
        boxes = boxes.cpu().numpy()
        scores = logits.squeeze(0).data.cpu().numpy()[:, 1]
        landmarks = decode_landmark(landmarks.data.squeeze(0), anchors, cfg.TRAIN.encode_variance)
        scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale = scale.to(device)
        landmarks = landmarks * scale
        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        # sort score with descending order
        order = scores.argsort()[::-1]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # nms
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets = dets[keep, :]
        landmarks = landmarks[keep]

        # keep top-K after NMS
        dets = dets[:args.keep_top_k, :]
        landmarks = landmarks[:args.keep_top_k, :]

        # concatenate boxes, scores and landmarks
        dets = np.concatenate((dets, landmarks), axis=1)
        timer['misc'].toc()

        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as f:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            f.write(file_name)
            f.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                f.write(line)

        print('img_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(
                i + 1, num_images, timer['forward_pass'].average_time, timer['misc'].average_time
            )
        )

        # save image
        if args.save_image:
            for det in dets:
                if det[4] < args.vis_threshold:  # if score is below visualize threshold
                    continue
                text = "{:.4f}".format(det[4])
                det = list(map(int, det))
                cv2.rectangle(img_bgr, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 2)
                cx = det[0]
                cy = det[1] + 12
                cv2.putText(img_bgr, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landmarks
                cv2.circle(img_bgr, (det[5], det[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_bgr, (det[7], det[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_bgr, (det[9], det[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_bgr, (det[11], det[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_bgr, (det[13], det[14]), 1, (255, 0, 0), 4)
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + str(i) + ".jpg"
            cv2.imwrite(name, img_bgr)

