import os, sys
sys.path.append('.')
import assemble
from configs.mtcnn_config import config


def assemble_pnet_anno_files(cfg, mode='train'):
    net_size = config.MODEL.PNET_SIZE

    pnet_postive_file = os.path.join(cfg.DATA.mtcnn_anno_path, 'pos_{}_{}.txt'.format(net_size, mode))
    pnet_part_file = os.path.join(cfg.DATA.mtcnn_anno_path, 'part_{}_{}.txt'.format(net_size, mode))
    pnet_neg_file = os.path.join(cfg.DATA.mtcnn_anno_path, 'neg_{}_{}.txt'.format(net_size, mode))
    imglist_filename = os.path.join(cfg.DATA.mtcnn_anno_path, 'imglist_anno_{}_{}.txt'.format(net_size, mode))

    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename, anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)


if __name__ == '__main__':
    assemble_pnet_anno_files(config, 'train')
    assemble_pnet_anno_files(config, 'val')

