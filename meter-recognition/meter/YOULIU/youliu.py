# -*- coding: utf-8 -*-
import cv2
import configure

from third_part.leinao_segmentation.api import LeiNaoUNet
from .utils import recogniton_state


class MethodYouLiuSGState(object):
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

class YouLiu(object):
    def __init__(self):
        if configure.config.name in ['youliu_1', 'youliu_2', 'youliu_3']:
            self.method = MethodYouLiuSGState()

    def __call__(self, img, filename, kp, seg):
        return self.method(img, filename, kp, seg)
