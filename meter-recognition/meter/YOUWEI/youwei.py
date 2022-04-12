# -*- coding: utf-8 -*-
import cv2
import configure

from third_part.hourglass.api import KeyPoint
from third_part.leinao_segmentation.api import LeiNaoUNet
from method.kp_seg1_old import MethodKpSegOneOld

from method.utils import improve_landmark_with_template
from .utils import recogniton_rectangle, recogniton_state, recogniton_youwei_reading


class MethodYouWeiSGState(object):
    def __init__(self):
        self.sg = LeiNaoUNet()

    def __call__(self, img, filename, kp, seg):
        result_list = []
        if seg is not None:
            img_seg = seg
            img_show = img.copy()
        else:
            img_seg, img_show = self.sg(img, configure.config.seg_resolution, 0, configure.config.seg_resolution)
        result_list.append(recogniton_state(configure.config.name, img_seg, None, configure.config.scale_carve,
                                            img_show, filename, configure.config.save_dir))
        return result_list


class MethodYouWeiKpSgRectangle(object):
    def __init__(self):
        self.kp = KeyPoint()
        self.sg = LeiNaoUNet()

    def __call__(self, img, filename, kp, seg):
        result_list = []
        if kp is not None:
            landmarks = kp
            img_adjust = img
        else:
            landmarks, img_adjust = self.kp(img, configure.config.adjust_resolution, 128, configure.config.mean_std)
            if eval(configure.config.temp_path) is not None:
                landmarks = improve_landmark_with_template(landmarks)

        h_adj, w_adj, _ = img_adjust.shape
        img_show = img_adjust.copy()
        for point in landmarks:
            cv2.circle(img_show, center=(int(point[0]), int(point[1])), radius=2, color=(0, 255, 0), thickness=-1)
        # cv2.imwrite(os.path.join(save_dir,filename[:-4] + '_point.jpg'), img_show)

        if seg is not None:
            img_seg = seg
        else:
            img_seg, _ = self.sg(img, configure.config.seg_resolution, 0, configure.config.adjust_resolution)
        result_list.append(recogniton_rectangle(configure.config.name, img_seg, landmarks, configure.config.scale_carve,
                                                img_show, filename, configure.config.save_dir))
        return result_list


class MethodYouWeiSgRead(object):
    def __init__(self):
        self.sg = LeiNaoUNet()

    def __call__(self, img, filename, kp, seg):
        result_list = []
        if seg is not None:
            img_seg = seg
            img_show = img.copy()
        else:
            img_seg, img_show = self.sg(img, configure.config.seg_resolution, 0, configure.config.seg_resolution)
        result_list.append(recogniton_youwei_reading(configure.config.name, img_seg, None, configure.config.scale_carve,
                                                     img_show, filename, configure.config.save_dir))
        return result_list


class MethodYouWeiKpSgRead(object):
    def __init__(self):
        self.kp = KeyPoint()
        self.sg = LeiNaoUNet()

    def __call__(self, img, filename, kp, seg):
        result_list = []
        if kp is not None:
            landmarks = kp
            img_adjust = img.copy()
        else:
            landmarks, img_adjust = self.kp(img, configure.config.adjust_resolution, 128, configure.config.mean_std)
            landmarks = landmarks[:len(configure.config.scale_carve)]
            if eval(configure.config.temp_path) is not None:
                landmarks = improve_landmark_with_template(landmarks)

        h_adj, w_adj, _ = img_adjust.shape
        img_show = img_adjust.copy()
        for point in landmarks:
            cv2.circle(img_show, center=(int(point[0]), int(point[1])), radius=2, color=(0, 255, 0),
                       thickness=-1)
        if seg is not None:
            img_seg = seg
        else:
            img_seg, _ = self.sg(img, configure.config.seg_resolution, 0, configure.config.adjust_resolution)
        result_list.append(recogniton_youwei_reading(configure.config.name, img_seg, landmarks,
                                                     configure.config.scale_carve, img_show, filename,
                                                     configure.config.save_dir))
        return result_list


class MethodYouWeiKpSgState(object):
    def __init__(self):
        self.kp = KeyPoint()
        self.sg = LeiNaoUNet()

    def __call__(self, img, filename, kp, seg):
        result_list = []
        if kp is not None:
            landmarks = kp
            img_adjust = img.copy()
        else:
            landmarks, img_adjust = self.kp(img, configure.config.adjust_resolution, 128, configure.config.mean_std)

            if eval(configure.config.temp_path) is not None:
                landmarks = improve_landmark_with_template(landmarks)

        h_adj, w_adj, _ = img_adjust.shape
        img_show = img_adjust.copy()
        for point in landmarks:
            cv2.circle(img_show, center=(int(point[0]), int(point[1])), radius=2, color=(0, 255, 0),
                       thickness=-1)

        if seg is not None:
            img_seg = seg
        else:
            img_seg, _ = self.sg(img, configure.config.seg_resolution, 0, configure.config.adjust_resolution)
        result_list.append(recogniton_state(configure.config.name, img_seg, landmarks, configure.config.scale_carve,
                                            img_show, filename, configure.config.save_dir))
        return result_list


class YouWei(object):
    def __init__(self):
        if configure.config.name in ['52_bj', 'youwei_1']:
            self.method = MethodKpSegOneOld()
        if configure.config.name in ['youwei_2', 'youwei_3', 'youwei_17', 'youwei_18', 'youwei_25', 'youwei_30',
                                     'youwei_32', 'youwei_33', 'youwei_34', 'youwei_35', 'wasi_1']:
            self.method = MethodYouWeiSGState()
        if configure.config.name in ['youwei_10', 'youwei_12']:
            self.method = MethodYouWeiKpSgRectangle()
        if configure.config.name in ['youwei_9']:
            self.method = MethodYouWeiSgRead()
        if configure.config.name in ['youwei_7', 'youwei_19', 'youwei_20', 'youwei_21', 'youwei_23', 'youwei_24',
                                     'youwei_28', '32yweij_YZF3', 'dianliu_dianya']:
            self.method = MethodYouWeiKpSgRead()
        if configure.config.name in ['youwei_11', 'youwei_13', 'youwei_14', 'youwei_15', 'youwei_16', 'youwei_22',
                                     'youwei_27', 'youwei_29', 'youwei_31']:
            self.method = MethodYouWeiKpSgState()

    def __call__(self, img, filename, kp, seg):
        return self.method(img, filename, kp, seg)
