import logging

import argparse

import PIL
import cv2
import numpy as np
import torch
from torchvision import transforms

from TransReID.config import cfg
from TransReID.datasets import make_dataloader
from TransReID.model import make_model

from TransReID.vit_rollout import VITAttentionRollout

from trackers.tracking_utils.timer import Timer
from TransReID.datasets.pigreid import PIGreID

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config_file", default=r"D:\zlw\OC_SORT-master\TransReID\configs\DukeMTMC\vit_base.yml",
                         help="path to config file", type=str)
    args = parser.parse_args()
    return args

class Extractor:
    def __init__(self, use_cuda=True):
        super(Extractor, self).__init__()
        args = get_args()
        '''
        cfg导入，log，数据集，model初始化
        '''
        if args.config_file != "":
            cfg.merge_from_file(args.config_file)
        dataset = PIGreID(root=cfg.DATASETS.ROOT_DIR)
        num_classes = dataset.num_train_pids
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids
        self.model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num = view_num)
        self.model.load_param(cfg.TEST.WEIGHT)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(cfg.TEST.WEIGHT))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # self.attention_rollout = VITAttentionRollout(self.model)

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (224,224)
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        im_batch = torch.cat([self.transform(PIL.Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            # mask = self.attention_rollout(im_batch)
            mask = self.model(im_batch)
        return mask.cpu().numpy()


def _get_features(extractor, xyxy, ori_img):
    im_crops = []
    time2 = Timer()
    # time2.tic()
    for box in xyxy:
        x1, y1, x2, y2 = box
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, ori_img.shape[1])
        y2 = min(y2, ori_img.shape[0])
        im = ori_img[int(y1):int(y2), int(x1):int(x2)]
        im_crops.append(im)
    # time2.toc()
    # print("for循环取box的时间：", time2.diff)
    if im_crops:
        # time2.tic()
        features = extractor(im_crops)
        # time2.toc()
        # print("transformer提取特征：", time2.diff)
    else:
        features = np.array([])
    return features

