import json
import numpy as np
import PIL.Image
import PIL.ImageDraw
import cv2
from cfg import bjds_dict, seg_dict
import albumentations as A

def tf(img):
    # tf = A.CoarseDropout(max_holes=30, min_holes=10, max_height=20, max_width=20, min_height=5, min_width=5,
    #                      fill_value=100, always_apply=False, p=1)  # 随机擦除
    # tf = A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.08, always_apply=False, p=1)  # 模拟图像雾
    # tf = A.RandomBrightness(limit=0.9, always_apply=False, p=1)  # 随机亮度变化
    # tf = A.GaussianBlur(p=1)  # 使用随机大小的内核模糊输入图像。
    # tf = A.RandomRain(p=1)
    # tf = A.RandomSunFlare(src_radius=50, num_flare_circles_lower=6, num_flare_circles_upper=10, p=1)
    # tf = A.RandomShadow(p=1)
    # tf = A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)
    # tf = A.GaussNoise(var_limit=(900.0, 900.0), p=1)
    # tf = A.Compose([A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.08, always_apply=False, p=0.2),
    #                 A.RandomRain(p=0.2),
    #                 A.RandomSunFlare(p=0.2)])
    # img = tf(image=img)['image']
    return img

def load_json(img_path, json_path):
    img = cv2.imread(img_path)

    f = open(json_path, encoding='utf-8')
    try:
        json_info = json.load(f)
    except:
        print(f)
        return None, None, None

    shapes = json_info['shapes']
    shapes_copy = shapes.copy()
    height = json_info['imageHeight']
    width = json_info['imageWidth']

    crop_img_list = []
    label_list = []
    kp_list = []
    # crop_seg_list = []

    for shape in shapes:
        label = shape['label']
        # shape_type = shape['shape_type']
        one_kp_list = []

        mask = np.zeros((height, width), dtype=np.uint8)
        mask = PIL.Image.fromarray(mask, mode='P')
        draw = PIL.ImageDraw.Draw(mask)

        if label in bjds_dict.keys():
            coord = shape['points']
            x1 = min(max(int(float(coord[0][0])), 0), width - 1)
            y1 = min(max(int(float(coord[0][1])), 0), height - 1)
            x2 = min(max(int(float(coord[1][0])), 0), width - 1)
            y2 = min(max(int(float(coord[1][1])), 0), height - 1)
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            crop_img = img[ymin:ymax, xmin:xmax, :]
            crop_img = tf(crop_img)

            crop_img_list.append(crop_img)
            label_list.append(label)

            for j in range(len(shapes_copy)):
                sub_shape = shapes_copy[j]
                sub_label = sub_shape['label']
                if sub_label == "point":
                    sub_coord = sub_shape['points']
                    sub_x = sub_coord[0][0]
                    sub_y = sub_coord[0][1]
                    if (x1 < sub_x < x2) and (y1 < sub_y < y2):
                        one_kp_list.append([int(sub_x - x1), int(sub_y - y1)])
            kp_list.append(np.array(one_kp_list))

    return crop_img_list, label_list, kp_list


def find_interval(scale_carve, value):
    assert isinstance(scale_carve, list), "scale_carve must be a list"
    interval = np.array(scale_carve)
    return np.digitize(value, interval)

