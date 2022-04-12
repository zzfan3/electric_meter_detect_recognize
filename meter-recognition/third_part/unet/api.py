# -*- coding: utf-8 -*-
"""
    加载不同的模型，获得inference的概率图
"""
import configure
import os
import torch
import numpy as np
import warnings
from PIL import Image
import cv2
from ..common import letter_box_image
from .utils import Compose, SegToTensor
from .model import OUNet5, DUNet5
from collections import  OrderedDict

warnings.filterwarnings('ignore')


class DSegmentor(object):
    def __init__(self, segment_model_path=False):
        super(DSegmentor, self).__init__()
        if not segment_model_path:
            segment_model_path = configure.config.segment_model_path
        self.checkpoint_dir = segment_model_path
        self.gpu_num = 1
        self.transform = Compose([SegToTensor()])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DUNet5(3, 1)
        if self.gpu_num == 0:
            self.cuda = False
        else:
            self.cuda = True
        if os.path.exists(self.checkpoint_dir):
            # self.model = torch.jit.load(self.checkpoint_dir).to(self.device)
            model_param = torch.load(self.checkpoint_dir)
            new_weights  = OrderedDict()
            for name, weights in model_param.items():
                if 'module.' in name:
                    new_weights[name.replace('module.', '')] = weights
                else:
                    new_weights[name] = weights
            
            self.model.load_state_dict(new_weights)
            self.model = self.model.to(self.device)
            print("Load UNet Segment pretrained model success!")
        else:
            assert False, "Load UNet Segment pretrained model fail!"

    def preprocess(self, img):
        # BGR->RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        # 保持长宽比的resize
        img = letter_box_image(img, resolution_seg, resolution_seg, value)
        self.model.eval()
        with torch.no_grad():
            img_tensor = self.preprocess(img)
            img_tensor = img_tensor.to(self.device)
            outputb, outputr = self.model(img_tensor)
            # 黑色指针
            out_npyb = (outputb.data.cpu().numpy())
            out_npyb = np.squeeze(out_npyb)
            out_imgb = Image.fromarray((out_npyb * 255).astype(np.uint8))
            ###################################
            out_binaryb = np.zeros(out_npyb.shape)
            out_binaryb[np.array(out_imgb) > 127] = 255
            out_binaryb = cv2.resize(out_binaryb, (resolution_adj, resolution_adj))
            # out_binary_imgb=Image.fromarray(out_binaryb.astype(np.uint8))
            # 红色指针
            out_npyr = (outputr.data.cpu().numpy())
            out_npyr = np.squeeze(out_npyr)
            out_imgr = Image.fromarray((out_npyr * 255).astype(np.uint8))
            ###################################
            out_binaryr = np.zeros(out_npyr.shape)
            out_binaryr[np.array(out_imgr) > 127] = 255
            out_binaryr = cv2.resize(out_binaryr, (resolution_adj, resolution_adj), interpolation=cv2.INTER_NEAREST)
            # out_binary_imgr = Image.fromarray(out_binaryr.astype(np.uint8))
            return out_binaryb, out_binaryr


class OSegmentor(object):
    def __init__(self, segment_model_path=False):
        super(OSegmentor, self).__init__()
        if not segment_model_path:
            segment_model_path = configure.config.segment_model_path
        self.checkpoint_dir = segment_model_path
        self.gpu_num = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = Compose([SegToTensor()])
        self.model = OUNet5(3, 1)
        if self.gpu_num == 0:
            self.cuda = False
        else:
            self.cuda = True
        if os.path.exists(self.checkpoint_dir):
            # self.model.load_state_dict(torch.load(self.checkpoint_dir).to(self.device))
            # self.model = self.model.load_state_dict(torch.load(self.checkpoint_dir).to(self.device))
            model_param = torch.load(self.checkpoint_dir)
            new_weights  = OrderedDict()
            for name, weights in model_param.items():
                if 'module.' in name:
                    new_weights[name.replace('module.', '')] = weights
                else:
                    new_weights[name] = weights
            
            self.model.load_state_dict(new_weights)
            ## self.model.load_state_dict(model_param)
            self.model = self.model.to(self.device)
            print("Load Segment pretrained model success!")
        else:
            assert False, "Load Segment pretrained model fail!"

    def preprocess(self, img):
        # BGR->RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        # 保持长宽比的resize
        img = letter_box_image(img, resolution_seg, resolution_seg, value)
        self.model.eval()
        with torch.no_grad():
            img_tensor = self.preprocess(img)
            img_tensor = img_tensor.to(self.device)
            output, _, _, _ = self.model(img_tensor)
            out_npy = (output.data.cpu().numpy())
            out_npy = np.squeeze(out_npy)  #
            out_img = Image.fromarray((out_npy * 255).astype(np.uint8))  #
            out_binary = np.zeros(out_npy.shape)
            out_binary[np.array(out_img) > 127] = 255
            # cv2.imwrite(os.path.join(configure.config.save_dir,'img_segment.jpg'),out_binary)
            out_binary_img = cv2.resize(out_binary, (resolution_adj, resolution_adj), interpolation=cv2.INTER_NEAREST)  # 分割结果resize到256
            return out_binary_img, _


if __name__ == "__main__":
    # 设置模型或模型列表     【必须】
    model_list = [r'weights/JCQ_3A_20_UNet5_aug_py3.6_torch_1.3.0_dice_0.9308.pth']
    # 设置待处理图像的文件夹 【必须】
    input_dir = r'data/test_code'
    file_list = os.listdir(input_dir)
    file_path_list = [os.path.join(input_dir, file) for file in file_list]
    # 设置预处理
    transform = Compose([SegToTensor()])
    # 设置计算设备cuda或cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 逐个模型进行inference
    for model_path in model_list:
        # 加载模型
        if model_path.endswith('_jit.pt'):
            model = torch.jit.load(model_path).to(device)
        else:
            model = torch.load(model_path).to(device)
        # 设置保存概率图文件夹
        save_dir_probability = input_dir + '_probability'
        if not os.path.exists(save_dir_probability):
            os.makedirs(save_dir_probability)
        # 设置保存二值化图文件夹
        save_dir_binary = input_dir + '_binary'
        if not os.path.exists(save_dir_binary):
            os.makedirs(save_dir_binary)
        # 逐个图像处理
        model.eval()
        with torch.no_grad():
            for i, file_path in enumerate(file_path_list):
                img = Image.open(file_path)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                print('file_path:', file_path)
                img_tensor = transform(img)
                img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)
                output, _, _, _ = model(img_tensor)
                out_npy = (output.data.cpu().numpy())
                out_npy = np.squeeze(out_npy)
                out_img = Image.fromarray((out_npy * 255).astype(np.uint8))
                save_path_probability = os.path.join(save_dir_probability, os.path.split(file_path)[1])
                # 保存概率图
                out_img.save(save_path_probability)
                ###################################
                out_binary = np.zeros(out_npy.shape)
                out_binary[np.array(out_img) > 127] = 255
                out_binary_img = Image.fromarray(out_binary.astype(np.uint8))
                save_path_binary = os.path.join(save_dir_binary, os.path.split(file_path)[1][:-4] + '_binary.jpg')
                # 保存二值化图
                out_binary_img.save(save_path_binary)
                del out_binary_img, out_img, img_tensor
        del model
