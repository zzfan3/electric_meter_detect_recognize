# -*- coding: utf-8 -*-
import cv2
import configure
from third_part.hourglass.api import KeyPoint

from method.kp_seg1_old import MethodKpSegOneOld
from .utils import process_no_segreg


class MethodSF6Tmp(object):
    def __init__(self):
        self.kp = KeyPoint()

    def __call__(self, img, filename, kp, seg):
        result_list = []
        if kp is not None:
            landmarks = kp
            img_adjust = img
        else:
            landmarks, img_adjust = self.kp(img, configure.config.adjust_resolution, 128, configure.config.mean_std)
        h_adj, w_adj, _ = img_adjust.shape
        img_show = img_adjust.copy()
        for point in landmarks:
            cv2.circle(img_show, center=(int(point[0]), int(point[1])), radius=2, color=(0, 0, 255), thickness=-1)
        # cv2.imwrite(os.path.join(save_dir,filename[:-4] + '_point.jpg'), img_show)
        result_list.append(process_no_segreg(img_adjust, landmarks, filename))
        return result_list


class SF6(object):
    def __init__(self):
        if not configure.config.segment:
            self.method = MethodSF6Tmp()
        else:
            self.method = MethodKpSegOneOld()

    def __call__(self, img, filename, kp, seg):
        if configure.config.name in ['12_SF6']:
            img = cv2.rotate(img, cv2.ROTATE_180)
        return self.method(img, filename, kp, seg)
