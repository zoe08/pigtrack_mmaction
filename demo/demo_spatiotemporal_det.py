# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import os.path as osp
import shutil
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import load_checkpoint

from mmaction.models import build_detector
from mmaction.utils.decorators import import_module_error_func, Timer

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.datasets import COCO_CLASSES

from tracker.pigstrack.ocsort import OCSort

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):

    @import_module_error_func('mmdet')
    def inference_detector(*args, **kwargs):
        pass

    @import_module_error_func('mmdet')
    def init_detector(*args, **kwargs):
        pass


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR,  (255, 255, 255)white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    # color = ((30 * idx) % 255, (10 * idx) % 255, (50 * idx) % 255)

    return color

def visualize(frames, annotations, plate=plate_blue, max_num=5):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted results.
        plate (str): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.

    Returns:
        list[np.ndarray]: Visualized frames.
    """
    result = []

    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])
    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]
            for ann in anno:
                box = ann[0][:4]
                tid = ann[0][4]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                color = get_color(abs(tid))
                cv2.rectangle(frame, st, ed,  color, 2)  # 画检测框
                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = 'ID:' + str(int(tid))
                    text = ' '.join([text, abbrev(lb)])
                    text = ':'.join([text, '{:.2f}'.format(score[k])])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 25)
                    diag1 = (location[0], location[1])
                    cv2.rectangle(frame, diag0, diag1, color, -1)  # 画text背景
                    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, FONTSCALE,
                                FONTCOLOR, 2, LINETYPE)
                    result.append(
                            f"{ind} {tid} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f} {lb}\n")

    return frames_, result



def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--config',
        default=(r'D:\zlw\mmaction2-0.22.0\configs\detection\ava\slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py'),
        help='spatio temporal detection config file path')
    parser.add_argument(
        '--checkpoint',
        default=(r'D:\zlw\mmaction2-0.22.0\tools\test\epoch_20.pth'),
        help='spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default=r'D:\zlw\mmaction2-0.22.0\mmdetection-2.20.0\configs\yolox\yolox_m_8x8_300e_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=(r'D:\zlw\YOLOX-main\YOLOX_outputs\_pig_coco_action\epoch_10_ckpt.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--label-map',
        default=r'D:\zlw\mmaction2-0.22.0\tools\data\ava\label_map.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--det-score-thr', type=float, default=0.7, help='the threshold of human detection score')
    parser.add_argument('--action-score-thr',type=float,default=0.4,help='the threshold of human action score')  # 0.4 不框选总比错框好
    parser.add_argument('--video',default=r'D:\zlw\_video\20.mp4', help='video file/url') # r'D:\zlw\_video\5.mp4'
    parser.add_argument('--out-filename', default=r'D:\zlw\_long_video\20_pt_demo.mp4', help='output filename')
    parser.add_argument('--output',default=r"D:\zlw\mmaction2-0.22.0\demo\20_pt.txt",  help='output txt')
    parser.add_argument('--predict-stepsize',default=1, type=int,help='give out a prediction per n frames') # 8 , 1
    parser.add_argument('--output-stepsize', default=1, type=int,help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0')) # 4 , 1
    parser.add_argument('--output-fps', default=6,type=int,help='the fps of demo video output')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def frame_extraction(video_path):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
    return frame_paths, frames

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device=torch.device("cuda"),
        fp16=False
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        '''
        :param img:
        :param timer:
        :return: output
        '''
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        # 原有代码
        # img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)  #
        # 修改代码
        img2, img, ratio = preproc(img, self.test_size)

        img_info["ratio"] = ratio
        img2 = torch.from_numpy(img2).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img2)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            # nms 相当于后处理
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre)
            # timer.toc()
            # TODO 加速
            # logger.info("yolox目标检测花费时间：{} fps".format(1/ timer.duration))
        return outputs, img_info

def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    # model = init_detector(args.det_config, args.det_checkpoint, args.device)
    # assert model.CLASSES[0] == 'pig', ('We require you to use a detector '
    #                                       'trained on COCO')
    #TODO yoloXmodel
    exp = get_exp(r'D:\zlw\OC_SORT-master\exps\default\yolox_m.py')
    model = exp.get_model().to(args.device)
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()
    ckpt_file = args.det_checkpoint
    # logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    # logger.info("loaded checkpoint done.")
    predictor = Predictor(model, exp, COCO_CLASSES)
    timer = Timer()
    results = []
    print('Performing pig Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    tracker = OCSort(det_thresh=0.3, iou_threshold=0.5)
    for frame_path in frame_paths:
        output,img_info = predictor.inference(frame_path, timer)
        # TODO tracker
        online_targets = tracker.update(output[0], img_info, exp.test_size)
        result = torch.tensor(online_targets[:, :5], dtype=torch.float32)
        # We only keep human detections with score larger than det_score_thr
        # result = result[result[:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.

    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:,  0:4:2] /= img_w
    human_detection[:,  1:4:2] /= img_h
    results, lines = [], []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results


def main():
    args = parse_args()

    frame_paths, original_frames = frame_extraction(args.video)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # resize frames to shortside 256 resize帧
    new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
    frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    w_ratio, h_ratio = new_w / w, new_h / h

    # Get clip_len, frame_interval and calculate center index of each clip
    config = mmcv.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    val_pipeline = config.data.val.pipeline

    sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    # Note that it's 1 based here
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)  # 中心帧 [32,40 48 56 ..1768]

    # Load label_map
    label_map = load_label_map(args.label_map)
    try:
        if config['data']['train']['custom_classes'] is not None:
            label_map = {
                id + 1: label_map[cls]
                for id, cls in enumerate(config['data']['train']
                                         ['custom_classes'])
            }
    except KeyError:
        pass

    # Get Human detection results
    track = []
    center_frames = [frame_paths[ind - 1] for ind in timestamps]  # 中心帧路径
    human_detections = detection_inference(args, center_frames)  # 中心帧的检测
    for i in range(len(human_detections)):
        det = human_detections[i]
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = det[:, :5].to(args.device)

    # Get img_norm_cfg 图像标准化
    img_norm_cfg = config['img_norm_cfg']
    if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
        to_bgr = img_norm_cfg.pop('to_bgr')
        img_norm_cfg['to_rgb'] = to_bgr
    img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
    img_norm_cfg['std'] = np.array(img_norm_cfg['std'])

    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        config['model']['test_cfg']['rcnn']['action_thr'] = .0
    except KeyError:
        pass

    config.model.backbone.pretrained = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))

    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    predictions, cls_ids = [], []
    results = []
    print('Performing SpatioTemporal Action Detection for each clip')

    assert len(timestamps) == len(human_detections)
    prog_bar = mmcv.ProgressBar(len(timestamps))
    for timestamp, proposal in zip(timestamps, human_detections):
        proposal = proposal[:,:4]
        if proposal.shape[0] == 0:
            predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        imgs = [frames[ind].astype(np.float32) for ind in frame_inds]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(args.device)

        with torch.no_grad():
            result = model(
                return_loss=False,
                img=[input_tensor],
                img_metas=[[dict(img_shape=(new_h, new_w))]],
                proposals=[[proposal]])
            result = result[0]
            prediction, cls_id = [], []
            # N proposals
            for i in range(proposal.shape[0]):
                prediction.append([])
                # cls_id.append([])
            # Perform action score thr
            for i in range(len(result)):
                if i + 1 not in label_map:
                    continue
                for j in range(proposal.shape[0]):
                    if result[i][j, 4] > args.action_score_thr:
                        prediction[j].append((label_map[i + 1], result[i][j, 4]))
                    #     cls_id[j].append(i + 1)
                    # else:
                    #     cls_id[j] = 0  #
            predictions.append(prediction)
            # cls_ids.append(cls_id)
        prog_bar.update()

    for human_detection, prediction in zip(human_detections, predictions):
        results.append(pack_result(human_detection, prediction, new_h, new_w))


    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int)

    dense_n = int(args.predict_stepsize / args.output_stepsize)
    frames = [
        cv2.imread(frame_paths[i - 1])
        for i in dense_timestamps(timestamps, dense_n)
    ]
    print('Performing visualization')
    vis_frames, result = visualize(frames, results)  # TODO 加入轨迹ID
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                fps=args.output_fps)  # 6
    vid.write_videofile(args.out_filename)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)

    # 存储数据
    ava_file = args.output
    with open(ava_file, 'w') as f:
        f.writelines(result)


if __name__ == '__main__':
    main()
