# -*- coding: utf-8 -*-
import os
import torch
from datetime import datetime
import configparser


class Configure(object):
    def __init__(self):
        pass

    def set_arg(self, configpath=None, save=True):
        super(Configure, self).__init__()
        assert os.path.exists(configpath)
        self.cf = configparser.ConfigParser()
        self.cf.read(configpath, encoding="utf-8")

        assert self.cf.has_section("bjds")
        # string
        self.name = self.get('bjds', 'name')  # 表计名称
        self.temp_path = self.get('bjds', 'temp_path')  # 模板json的路径
        self.segment_model_path = self.get('bjds', 'segment_model_path')  # 分割模型路径
        self.adjust_model_path = self.get('bjds', 'adjust_model_path')  # 特征点定位模型路径
        self.test_dir = self.get('bjds', 'test_dir')  # 测试数据集
        self.save_dir = self.get('bjds', 'save_dir')  # 保存文件夹
        # bool
        self.keypoint = self.cf.getboolean('bjds', 'keypoint')  # 是否使用hourglass关键点方案
        self.segment = self.cf.getboolean('bjds', 'segment')  # 是否使用Unet分割方案
        self.mean_std = self.cf.getboolean('bjds', 'mean_std')  # 特征点匹配时，是否进行去均值除方差
        # int
        self.flag = self.getInt('bjds', 'flag')  # 表计异常值处理类型
        self.adjust_resolution = self.getInt('bjds', 'adjust_resolution')  # 特征点定位图像分辨率
        self.seg_resolution = self.getInt('bjds', 'seg_resolution')  # 指针分割图像分辨率
        self.pointer_num = self.getInt('bjds', 'pointer_num')  # 表计指针数据
        self.radius_out_b = self.getInt('bjds', 'radius_out_b')  # 指针1外圆半径
        self.radius_out_r = self.getInt('bjds', 'radius_out_r')  # 指针2外圆半径
        self.radius_in_b = self.getInt('bjds', 'radius_in_b')  # 指针1内圆半径
        self.radius_in_r = self.getInt('bjds', 'radius_in_r')  # 指针2内圆半径
        self.feature_point_num = self.getInt('bjds', 'feature_point_num')  # 特征点数目
        # float
        self.point_level = self.getFloat('bjds', 'point_level')  # 指针水平时对应的刻度
        # list float
        self.scale_carve = self.getListFloat('bjds', 'scale_carve')  # 表计主刻度

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置运算设备
        if save:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

    def get(self, section, option):
        val = self.cf.get(section, option).strip('"')
        if val == "None":
            return val
        else:
            return val

    def getBool(self, section, option):
        val = self.get(section, option).strip()
        if val == "None":
            return False
        else:
            return self.cf.getboolean(section, option)

    def getInt(self, section, option):
        val = self.get(section, option).strip()
        if val == "None":
            return 0
        else:
            return self.cf.getint(section, option)

    def getFloat(self, section, option):
        val = self.get(section, option).strip()
        if val == "None":
            return 0.0
        else:
            return self.cf.getfloat(section, option)

    def getList(self, section, option):
        val = self.get(section, option).strip()
        if val == "None":
            return []
        else:
            return list(x.strip() for x in val.split(","))

    def getListFloat(self, section, option):
        return list(float(x) for x in self.getList(section, option))


config = Configure()


def main():
    print("name: ", config.name, type(config.name))
    print("flag: ", config.flag, type(config.flag))
    print("adjust_resolution: ", config.adjust_resolution, type(config.adjust_resolution))
    print("seg_resolution: ", config.seg_resolution, type(config.seg_resolution))
    print("segment: ", config.segment, type(config.segment))
    print("mean_std: ", config.mean_std, type(config.mean_std))
    print("pointer_num: ", config.pointer_num, type(config.pointer_num))
    print("scale_carve: ", config.scale_carve, type(config.scale_carve))
    print("temp_path: ", config.temp_path, type(config.temp_path))
    print("point_level: ", config.point_level, type(config.point_level))
    print("radius_out_b: ", config.radius_out_b, type(config.radius_out_b))
    print("radius_out_r: ", config.radius_out_r, type(config.radius_out_r))
    print("radius_in_b: ", config.radius_in_b, type(config.radius_in_b))
    print("radius_in_r: ", config.radius_in_r, type(config.radius_in_r))
    print("feature_point_num: ", config.feature_point_num, type(config.feature_point_num))
    print("segment_model_path: ", config.segment_model_path, type(config.segment_model_path))
    print("adjust_model_path: ", config.adjust_model_path, type(config.adjust_model_path))
    print("test_dir: ", config.test_dir, type(config.test_dir))
    print("save_dir: ", config.save_dir, type(config.save_dir))


if __name__ == "__main__":
    main()
