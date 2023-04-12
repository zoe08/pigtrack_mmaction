
import errno
import logging
import warnings

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os.path as osp
import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger

GRID_SPACING = 10

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument( "--config_file", default=r"./configs/DukeMTMC/vit_base.yml",
                         help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--image_path', type=str, default=r'D:\zlw\_PIGREID\query',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")
    return args

def check_isfile(fpath):
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile

def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    args = get_args()
    '''
    cfg导入，log，数据集，model初始化
    '''
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)

    logger = logging.getLogger("transreid.explain")
    logger.info("Enter inferencing")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    save_dir = './logs/duke_vit_base2'

    for name in os.listdir(args.image_path):
        image_path = os.path.join(args.image_path, name)
        output_path = os.path.join(save_dir, name)
        pid = name.split('_')[0]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        img = Image.open(image_path)
        H, W = img.size
        img = img.resize((224, 224))
        input_tensor = transform(img).unsqueeze(0)
        if args.use_cuda:
            input_tensor = input_tensor.cuda()

        if args.category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
                discard_ratio=args.discard_ratio)
            mask = attention_rollout(input_tensor)
            Name = "attention_rollout_{:.3f}_{}_{}.png".format(args.discard_ratio, args.head_fusion, pid)
            output_path1 = os.path.join(save_dir, 'actmap', Name)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
            mask = grad_rollout(input_tensor, args.category_index)
            Name = "grad_rollout_{}_{}_{:.3f}_{}.png".format(args.category_index,args.discard_ratio, args.head_fusion, pid)
            output_path1 = os.path.join(save_dir, 'actmap', Name)


        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)

        # np_img.resize(W, H)
        mask = cv2.resize(mask, (W, H))
        cv2.imshow("Input Image", np_img)
        cv2.imshow(Name, mask)

        cv2.imwrite(output_path, np_img)
        cv2.imwrite(output_path1, mask)
        # cv2.waitKey(-1)