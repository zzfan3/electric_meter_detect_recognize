import os
import cv2
import math
import numpy as np

def recogniton_state(name, img_seg, landmarks, scale, img_show, filename, save_dir):
    value = -255    # 表计读数值
    state_value = -255   # 油位低：-1；油位正常：0；油位高：1

    # 纯分割只有指针
    if name in ["youliu_1", "youliu_2", "youliu_3"]:
        value = youliu_1_postprocess(img_seg, img_show)
    if value == -255:
        # print('关键点或者指针定位失败，请进行人工识别！')
        cv2.imwrite(os.path.join(save_dir, filename), img_show)
    else:
        # print(value, end=" ")
        # scale[0]为配置文件中的可配置参数
        if scale[0]:
            state_value = 1 if value > 0 else 0
        else:
            state_value = 0 if value > 0 else 1
        cv2.putText(img_show, str(state_value) + ": " + str(value), (20, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(save_dir, filename), img_show)
    return state_value

def youliu_1_postprocess(img_seg, img_show):
    # 指针分割
    img_seg = np.array(img_seg[0], np.uint8)
    angle = pointer_angle(img_seg, img_show)
    return angle

def pointer_angle(img_seg, img_show):
    angle = -255
    # 查找指针分割轮廓
    contours = findsortContours(img_seg)
    # 没有分割出指针区域
    if len(contours) == 0:
        return angle
    cv2.drawContours(img_show, contours[0], -1, (255, 0, 0), 1)
    
    [vx, vy, x, y] = cv2.fitLine(contours[0], cv2.DIST_L12, 0, 0.01, 0.01)
    angle = math.atan(vy/vx)
    angle = int(angle * 180 / math.pi)
    return angle

# 查找分割轮廓并排序
def findsortContours(img_seg):
    # 二值化
    ret, thresh = cv2.threshold(img_seg, 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    if cv2.__version__.startswith('4'):
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif cv2.__version__.startswith('3'):
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        raise AssertionError('cv2 must be either version 3 or 4 to call this method')
    # 按轮廓面积对轮廓进行排序
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    return contours