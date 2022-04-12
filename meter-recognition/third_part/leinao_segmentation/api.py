# -*- coding: utf-8 -*-
"""
    加载不同的模型，获得inference的概率图
"""
# from __future__ import print_function, division
import configure
import os
import torch
import numpy as np
import warnings
import cv2
from .model import UNet2x
from ..common import letter_box_image
from collections import  OrderedDict

warnings.filterwarnings('ignore')


class LeiNaoUNet(object):
    def __init__(self, segment_model_path=False, pointer_num=False):
        super(LeiNaoUNet, self).__init__()
        if not segment_model_path:
            segment_model_path = configure.config.segment_model_path
        if not pointer_num:
            pointer_num = configure.config.pointer_num
        self.checkpoint_dir = segment_model_path
        self.pointer_num = pointer_num
        self.gpu_num = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # output feature maps number = configure.config.pointer_num + 1
        self.model = UNet2x(3, self.pointer_num + 1).to(self.device)
        if self.gpu_num == 0:
            self.cuda = False
        else:
            self.cuda = True
        if os.path.exists(self.checkpoint_dir):
            # self.model = torch.jit.load(self.checkpoint_dir).to(self.device)
            model_param = torch.load(self.checkpoint_dir)
            # model_param = torch.load(self.checkpoint_dir, map_location=torch.device('cpu'))
            new_weights  = OrderedDict()
            for name, weights in model_param.items():
                if 'module.' in name:
                    new_weights[name.replace('module.', '')] = weights
                else:
                    new_weights[name] = weights
            
            self.model.load_state_dict(new_weights)
            #self.model.load_state_dict(model_param)
            self.model = self.model.to(self.device)
            print("Load LeiNao segment pretrained model success!")
        else:
            assert False, "Load LeiNao segment pretrained model fail!"

    def preprocess(self, img):
        # BGR->RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ##
        # [0,255]->[0,1]
        img = np.array(img).astype(np.float32)
        if len(img.shape) == 3:
            pass
        else:
            img = img[:, :, np.newaxis]
        # 减均值除方差
        mean = np.mean(img[img[..., 0] > 0], axis=0)
        std = np.std(img[img[..., 0] > 0], axis=0)
        pic = (img - mean) / (std + 1e-6)
        # image to tensor
        img_tensor = torch.from_numpy(pic.transpose((2, 0, 1))).unsqueeze_(0)
        return img_tensor

    def __call__(self, img, resolution_seg, value, resolution_adj, info=True):
        img = letter_box_image(img, resolution_seg, resolution_seg, value)
        self.model.eval()
        with torch.no_grad():
            img_tensor = self.preprocess(img)
            img_tensor = img_tensor.to(self.device)
            # 统一分割模型
            output, _, _, _ = self.model(img_tensor)  # torch.Size([1, n, size, size])
            output = (output.data.cpu().numpy())
            output = np.squeeze(output)
            predict = np.argmax(output, 0)
            res = []
            for i in range(1, self.pointer_num + 1):
                temp = np.zeros(predict.shape)
                temp[predict == i] = 255
                out_binary_img = cv2.resize(temp, (resolution_adj, resolution_adj), interpolation=cv2.INTER_NEAREST)
                res.append(out_binary_img)
            res = np.array(res)
        return res, img
