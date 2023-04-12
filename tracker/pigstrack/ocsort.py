"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
    OOS（Observation-centric Online Smoothing）：减少KF带来的累积误差
    OCM（Observation-centric Momentum）：在代价矩阵中加入轨迹的方向一致性，更好地实现匹配
    OCR（Observation-centric Recovery）：恢复由于遮挡造成的跟丢问题
"""
from __future__ import print_function

import numpy as np
import scipy
import torch
from collections import deque

from .association import *
from tracker.pigstrack.TransReID_extractor import Extractor, _get_features
# from tracker.tracking_utils.timer import Timer
from loguru import logger

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i  # 这样是倒序来找，先找前3帧的，如果没有前三帧，在依次递减，找前二帧。
        if cur_age - dt in observations:
            return observations[cur_age-dt]  # 找到就返回值
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold, budget=None):

        if metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    ** feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, score=None, cls=None, curr_feat=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.curr_feat = np.asarray(curr_feat, dtype=np.float32)
        self.score = score
        self.cls = cls

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xysr(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2  # xc,yc 都为 t l + w h /2
        ret[2] = ret[2]*ret[3]
        ret[3] = ret[2]/ret[3]  # a = w / h
        return ret

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2



class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, det=None, delta_t=3, initate=False, feat_history=50, orig=False):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
          from .kalmanfilter import KalmanFilterNew as KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                            0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.  # 观测噪声
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities, 不能观测到的初始动量给予高的不确定度
        self.kf.P *= 10.  # 协方差
        self.kf.Q[-1, -1] *= 0.01  # 过程噪声
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        if initate:
            KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

        self.state = TrackState.Tentative
        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        if det.curr_feat is not None:
            self.update_features(det.curr_feat)
        self.alpha = 0.95

        self.score = det.score
        self.cls = det.cls

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)  # 归一化 feat/二范数
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat  #
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def increment_age(self):
        self.age += 1  # 该track自出现以来的总帧数加1
        self.time_since_update += 1  # 该track自最近一次更新以来的总帧数加1

    def update(self, bbox, new_track):  # bbox
        """
        Updates the state vector with observed bbox.

        """
        #  得 previous_box 前3/2/1帧的box，若遮挡就是last_observation
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation 最后观测值；如果无遮挡，就是
                previous_box = None
                for i in range(self.delta_t):  # delta_t = 3
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:  # 新出现的观测值
                        previous_box = self.observations[self.age-dt]   #
                        break
                if previous_box is None:
                    previous_box = self.last_observation   #
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)  # 计算得xc，yc的运动方向向量
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
            if new_track.curr_feat is not None:
                self.update_features(new_track.curr_feat)
            if self.state == TrackState.Tentative and self.hits >= self.delta_t:
                self.state = TrackState.Confirmed
            self.score = new_track.score
            self.cls = new_track.cls
        else:
            self.kf.update(bbox)


    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()  # 预测得的pos什么格式？

        if(self.time_since_update > 0):
            self.hit_streak = 0

        self.increment_age()

        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def gating_distance(self, measurements):
        mean = np.dot(self.kf.H, self.kf.x).T
        covariance = self.kf.S
        '''only position'''
        mean, covariance = mean[:, :2], covariance[:2, :2]
        measurements = measurements[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance)  # 下三角阵
        d = measurements - mean  # detection - track
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)  # 解ax=b方程中的x, (假定a是一个上/下三角矩阵)
        squared_maha = np.sum(z * z, axis=0)  # 马氏距离
        return squared_maha


    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret


    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed



"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}


class OCSort(object):
    def __init__(self, det_thresh, max_age=30, min_hits=3, 
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False, with_reid=True):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        # KalmanBoxTracker.count = 0
        self.with_reid = with_reid
        self.Extractor = Extractor()

    def update(self, output_results, img_info, img_size):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        time1 = Timer()
        if output_results is None:
            return np.empty((0, 7))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            cls = output_results[:, 6]
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info['height'], img_info['width']
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes //= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)  # x1y1x2y2+scores
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        scores = scores[remain_inds]
        cls = cls[remain_inds]
        time1.tic()

        if self.with_reid:
            features_keep = _get_features(self.Extractor, bboxes, img_info['raw_img'])
            # time1.toc()
            # logger.info('提取特征时间：{}'.format(time1.diff))
            # time1.tic()
            detections = [Detection(KalmanBoxTracker.tlbr_to_tlwh(bboxes), score, cls, f) for
                          (bboxes, score, cls, f) in zip(bboxes, scores, cls, features_keep)]
            # time1.toc()
            # logger.info('detection类初始化时间：{}'.format(time1.diff))
        else:
            detections = [Detection(KalmanBoxTracker.tlbr_to_tlwh(bboxes), score, cls) for
                          (bboxes, score, cls) in zip(bboxes, scores, cls)]


        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            # tracker预测
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):  # 检查不合法元素
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # 将NaN无效值设为mask_
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 从KF估计的速度方向
        last_boxes = np.array([trk.last_observation for trk in self.trackers])

        # TODO 加级联匹配 tracker更新
        """
            First round of association
        """
        if self.with_reid:
            confirmed_tracks = [
                i for i, t in enumerate(self.trackers) if t.is_confirmed()]  # True

            unconfirmed_tracks = [
                i for i, t in enumerate(self.trackers) if not t.is_confirmed()]

            matched_a, unmatched_dets, unmatched_trks_a = embedding_distance(detections, self.trackers, confirmed_tracks)

            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_trks_a if
                self.trackers[k].time_since_update == 1]  # 未激活轨迹加刚刚没有匹配上
            unmatched_trks_a = [
                k for k in unmatched_trks_a if
                self.trackers[k].time_since_update != 1]  # 已经一些时间没有匹配上的

            udets = np.array([dets[i] for i in unmatched_dets])
            utrks = np.array([trks[i] for i in iou_track_candidates])

            k_observations = np.array(
                [k_previous_obs(self.trackers[i].observations, self.trackers[i].age, self.delta_t) for i in iou_track_candidates])
            velocities = np.array(
                [self.trackers[i].velocity if self.trackers[i].velocity is not None else np.array((0, 0)) for i in iou_track_candidates])

            matched_b, unmatched_dets, unmatched_trks_b = iou_OCM(
                udets, utrks, unmatched_dets, iou_track_candidates, self.iou_threshold, velocities, k_observations, self.inertia)
            matched = np.concatenate((matched_a, matched_b), axis=0)
            unmatched_trks = np.array(list(set(unmatched_trks_a + list(unmatched_trks_b))))

            # tracks = [
            #     i for i, t in enumerate(self.trackers)]
            # velocities = np.array(
            #     [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
            # last_boxes = np.array([trk.last_observation for trk in self.trackers])
            # k_observations = np.array(
            #     [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])
            # ious_dists = iou_OCM(dets, trks, self.iou_threshold, velocities, k_observations, self.inertia,
            #                      two_steps=False)
            # ious_dists_mask = (ious_dists > 0.5)  # 拒绝低重叠的Reid match
            # emb_dists = embedding_distance(detections, self.trackers,  confirmed_tracks, two_steps=False)
            # emb_dists[emb_dists > 0.25] = 1.0
            # emb_dists[ious_dists_mask] = 1.0
            # dists = np.minimum(ious_dists, emb_dists)
            # matched, unmatched_dets, unmatched_trks = associate(dists)

        else:
            velocities = np.array(
                [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
            last_boxes = np.array([trk.last_observation for trk in self.trackers])
            k_observations = np.array(
                [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])
            matched, unmatched_dets, unmatched_trks = iou_OCM(
                dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], detections[m[0]])

        """
            Second round of associaton by OCR 恢复丢失对象
        """
        # BYTE association ByteTrack的关联 没用到 self.use_byte=False
        # if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
        #     u_trks = trks[unmatched_trks]
        #     iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
        #     iou_left = np.array(iou_left)
        #     if iou_left.max() > self.iou_threshold:
        #         """
        #             NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
        #             get a higher performance especially on MOT17/MOT20 datasets. But we keep it
        #             uniform here for simplicity
        #         """
        #         matched_indices = linear_assignment(-iou_left)
        #         print("matched_indices:{}".format(matched_indices))
        #         to_remove_trk_indices = []
        #         for m in matched_indices:
        #             det_ind, trk_ind = m[0], unmatched_trks[m[1]]
        #             if iou_left[m[0], m[1]] < self.iou_threshold:
        #                 continue
        #             self.trackers[trk_ind].update(dets_second[det_ind, :], detections[det_ind])
        #             to_remove_trk_indices.append(trk_ind)
        #         unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                # print("matched_indices:{}".format(rematched_indices))
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :], detections[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None, None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], detections[i], delta_t=self.delta_t, initate=True)
            self.trackers.append(trk)
        i = len(self.trackers)
        # print(len(self.trackers))
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                    可选择最近的观测observation或预测值，没区别
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1], [trk.score], [trk.cls])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            # print(ret)
            return np.concatenate(ret)  # 输出ret box+id+score+cls
        return np.empty((0, 7))

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))


    @staticmethod
    def _xyxy_to_tlwh_array(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_tlwh
    @staticmethod
    def tlwh_to_xyxy(bbox_tlwh):
        if isinstance(bbox_tlwh, np.ndarray):
            bbox_xyxy = bbox_tlwh.copy()
        elif isinstance(bbox_tlwh, torch.Tensor):
            bbox_xyxy = bbox_tlwh.clone()
        bbox_xyxy[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_xyxy[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]


