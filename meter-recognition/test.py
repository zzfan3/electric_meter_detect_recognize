import os
import numpy as np
import cv2
import PIL.Image
from utils import load_json, find_interval
from cfg import bjds_dict

import sys
sys.path.insert(0, "./modules/Meter-Recognition")
from meter_recognition_api import Bjds2


TEST_DIR = '/home/dell/D/dell/fjw/practice/leinao/Hourglass_Train_New/data/youwei_22'
TEST_LIST = os.path.join(TEST_DIR, 'all.txt')

seg_img_path = '/home/dell/D/dell/fjw/practice/leinao/leinao_segmentation-master/data/youwei_14_21_22/target'

ini_path = "configs/bjds_youwei_22.ini"

save_path = 'results/test_0329/gaussnoise_900'
if not os.path.exists(save_path):
    os.makedirs(save_path)
f = open(os.path.join(save_path, 'result.txt'), 'w')

if __name__ == "__main__":
    # 初始化表计识别
    bjds_recognitions = {}
    result_dict = {}

    bjds = Bjds2(ini_path)
    bjds_recognitions[bjds.name] = bjds
    result_dict[bjds.name] = []

    img_dir = os.path.join(TEST_DIR, "JPEGImages")
    ann_dir = os.path.join(TEST_DIR, "Annotations")

    list_path = TEST_LIST
    assert os.path.exists(list_path), list_path + ' not found'
    fp = open(list_path, 'r')
    lines = fp.readlines()
    for line in lines:
        line = line.strip()
        img_name = line

        img_path = os.path.join(img_dir, line+'.jpg')
        ann_path = os.path.join(ann_dir, line+'.json')
        seg_path = os.path.join(seg_img_path, line+'.png')

        assert os.path.exists(img_path), img_path + ' not found'
        assert os.path.exists(ann_path), ann_path + ' not found'
        assert os.path.exists(seg_path), seg_path + ' not found'

        img_list, label_list, kp_list= load_json(img_path, ann_path)

        seg_img = cv2.imread(seg_path)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2GRAY)

        w, h = seg_img.shape[0], seg_img.shape[1]
        res = np.zeros((1, w, h))
        res[0][seg_img != 0] = 255


        seg_list = [res]

        for idx, crop_mat in enumerate(img_list):
            label = label_list[idx]
            kp = kp_list[idx]
            seg = seg_list[idx]
            if bjds_dict[label] in bjds_recognitions:
                meter_file_name = img_name + "_" + bjds_dict[label] + "_" + str(idx) + ".jpg"
                meter_file_name_gt = img_name + "_" + bjds_dict[label] + "_gt_" + str(idx) + ".jpg"
                values = bjds_recognitions[bjds_dict[label]].process(crop_mat, meter_file_name)
                values_gt = bjds_recognitions[bjds_dict[label]].process(crop_mat, meter_file_name_gt, kp, seg)


                print("bbox_label: ", label, end='\t')
                print("values: ", values, end='\t')
                print("values_gt: ", values_gt, end='\t')
                scale_carve = bjds_recognitions[bjds_dict[label]].scale_carve
                print("scale_carve: ", scale_carve)

                f.write("bbox_label: " + label + '\t' +
                        "values: " + str(values) + '\t' +
                        "values_gt: " + str(values_gt) + '\t' +
                        "scale_carve: " + str(scale_carve) + '\n')

                if -255 in values_gt:
                    continue

                one_result = []
                for result_idx in range(len(values_gt)):
                    value_gt = values_gt[result_idx]
                    value = values[result_idx]
                    inter = find_interval(scale_carve, value_gt)
                    if inter == 0:
                        interval_min = scale_carve[0]
                        interval_max = scale_carve[1]
                    elif inter >= len(scale_carve):
                        interval_min = scale_carve[inter-2]
                        interval_max = scale_carve[inter-1]
                    else:
                        interval_min = scale_carve[inter-1]
                        interval_max = scale_carve[inter]

                    if value == -255:
                        mse = interval_max - interval_min
                    else:
                        mse = abs(value - value_gt) / (interval_max - interval_min)
                    one_result.append(mse)
                result_dict[bjds_dict[label]].append(one_result)
            else:
                print(label, end='\t')
                print(label + " is not support, check " + bjds_dict[label] +
                      " ini folder or Meter_Recognition modules")

    for key in result_dict.keys():
        array = np.array(result_dict[key])
        if array.size:
            result_dict[key] = array.mean(axis=0).tolist()
    print(result_dict)
    f.write(str(result_dict))
    f.close()
