import os
import cv2
import math
import numpy as np


def recogniton_rectangle(name, img_seg, landmarks, scale, img_show, filename, save_dir):
    value = -255
    if name in ["youwei_10", "youwei_12"]:
        value = youwei_10_postprocess(img_seg, landmarks, scale, img_show)
    if value == -255:
        # print('关键点或者油位定位失败，请进行人工识别！')
        cv2.imwrite(os.path.join(save_dir, filename), img_show)
    else:
        # print(value)
        cv2.putText(img_show, str(value), (10, 10), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(save_dir, filename), img_show)
    return value


def recogniton_youwei_reading(name, img_seg, landmarks, scale, img_show, filename, save_dir):
    value = -255
    # 不带圆心识别
    if name in ['youwei_7', 'youwei_21', 'dianliu_dianya']:
        value = youwei_20_postprocess(img_seg, landmarks, scale, img_show, has_center=False)
    # 带圆心识别
    if name in ["youwei_20", 'youwei_23', 'youwei_24', 'youwei_28', '32yweij_YZF3']:
        value = youwei_20_postprocess(img_seg, landmarks, scale, img_show, has_center=True)
    if name in ['youwei_19']:
        value = youwei_19_postprocess(img_seg, landmarks, scale, img_show)
    # 扇形
    if name in ["youwei_9"]:
        value = youwei_9_postprocess(img_seg, img_show)
    if value == -255:
        # print('关键点或者指针定位失败，请进行人工识别！')
        cv2.imwrite(os.path.join(save_dir, filename), img_show)
    else:
        # print(value)
        cv2.putText(img_show, str(value), (10, 10), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(save_dir, filename), img_show)
    return value


def recogniton_state(name, img_seg, landmarks, scale, img_show, filename, save_dir):
    value = -255    # 表计读数值
    state_value = -255   # 油位低：-1；油位正常：0；油位高：1

    # 纯分割只有油位
    if name in ["youwei_2", 'youwei_30', 'youwei_33', 'youwei_34']:
        value = youwei_2_postprocess(img_seg, img_show, is_upper=True, is_minrect=True)
    if name in ['youwei_17', 'youwei_32', 'youwei_35']:
        value = youwei_2_postprocess(img_seg, img_show, is_lower=True, is_minrect=True)
    if name in ['wasi_1']:
        value = youwei_2_postprocess(img_seg, img_show, is_lower=True, is_rect=True)
    # 纯分割只有指针
    if name == "youwei_3":
        value = youwei_3_postprocess(img_seg, img_show)
    # 纯分割只有下界
    if name in ["youwei_18"]:
        value = youwei_18_postprocess(img_seg, img_show)
    # 纯分割有上下界
    if name in ['youwei_25']:
        value = youwei_25_postprocess(name, img_seg, img_show, is_upper=True)
    # 具有MIN和MAX的带圆心指针状态
    if name in ["youwei_14", "youwei_15", 'youwei_22']:
        value = youwei_14_postprocess(name, img_seg, landmarks, img_show)
    if name in ["youwei_11"]:
        value = youwei_14_postprocess(name, img_seg, landmarks, img_show, is_upper=True)
    # 关键点加分割
    if name in ['youwei_13', 'youwei_27', 'youwei_31']:
        value = youwei_16_postprocess(name, img_seg, landmarks, img_show, is_upper=True, is_minrect=True)
    if name in ['youwei_29']:
        value = youwei_16_postprocess(name, img_seg, landmarks, img_show, is_upper=True, is_rect=True)
    if name in ["youwei_16"]:
        value = youwei_16_postprocess(name, img_seg, landmarks, img_show, is_lower=True, is_minrect=True)
    if value == -255:
        # print('关键点或者指针定位失败，请进行人工识别！')
        cv2.imwrite(os.path.join(save_dir, filename), img_show)
    else:
        if 0:
        # print(value, end=" ")
            if value < scale[0]:
                state_value = -1
            elif value >= scale[0] and value <= scale[1]:
                state_value = 0
            else:
                state_value = 1
        else:
            value = max(0, min(1,value))
            state_value = value
        cv2.putText(img_show, str(state_value) + ": " + str(value), (20, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(save_dir, filename), img_show)
    return value


def youwei_2_postprocess(img_seg, img_show, is_upper=False, is_lower=False, is_rect=False, is_minrect=False):
    value = -255

    img_seg = np.array(img_seg[0], np.uint8)
    img_h, img_w = img_seg.shape
    # 油位框
    contours = findsortContours(img_seg)
    if len(contours) == 0:
        return value
    if is_rect:
        x, y, w, h = cv2.boundingRect(contours[0])
        # cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if is_lower:
            lpl = (x, y + h)
            lpr = (x + w, y + h)
        if is_upper:
            lpl = (x, y)
            lpr = (x + w, y)
    if is_minrect:
        # 油位最小矩形框
        rect, box = findminAreaRect(contours[0])
        cv2.drawContours(img_show, [box], 0, (0, 0, 255), 1)
        if is_lower:
            # 最小外接矩形下界左右端点
            if rect[0][0] < box[0][0]:
                lpl = box[1]
                lpr = box[0]
            else:
                lpl = box[0]
                lpr = box[3]
        if is_upper:
            if rect[0][0] < box[0][0]:
                lpl = box[2]
                lpr = box[3]
            else:
                lpl = box[1]
                lpr = box[2]
    cv2.line(img_show, (lpl[0], lpl[1]), (lpr[0], lpr[1]), (0, 255, 0), 1)
    # 取矩形框上方两端点中点为油位线
    yw_h = (lpl[1] + lpr[1]) // 2
    # 油位高度与表盘高度比值
    return 1 - yw_h / img_h 


def youwei_3_postprocess(img_seg, img_show):
    # 指针分割
    img_seg = np.array(img_seg[0], np.uint8)
    angle = pointer_angle(img_seg, img_show)
    return angle


def youwei_9_postprocess(img_seg, img_show):
    value = -255

    red_seg, white_seg = img_seg
    red_seg = np.array(red_seg, np.uint8)
    white_seg = np.array(white_seg, np.uint8)

    red_contours = findsortContours(red_seg)
    white_contours = findsortContours(white_seg)
    # 红白都分割不出来
    if len(red_contours) == 0 and len(white_contours) == 0:
        return value
    # 全白
    if len(red_contours) == 0:
        value = 1.0
        return value
    # 全红
    if len(white_contours) == 0:
        value = 0.0
        return value

    cv2.drawContours(img_show, red_contours[0], -1, (255, 0, 0), 1)
    red_area = cv2.contourArea(red_contours[0])
    cv2.drawContours(img_show, white_contours[0], -1, (0, 255, 0), 1)
    white_area = cv2.contourArea(white_contours[0])

    value = white_area / (white_area + red_area)

    return value


def youwei_10_postprocess(img_seg, landmarks, scale, img_show):
    value = -255
    # 判断估计的关键点坐标是不是递增的
    for i in range(len(landmarks) - 1):
        if landmarks[i][1] < landmarks[i + 1][1]:
            return value
    img_seg = np.array(img_seg[0], np.uint8)
    # 查找油位轮廓并已经按照轮廓面积大小排序
    contours = findsortContours(img_seg)
    if len(contours) == 0:
        value = scale[0]
        return value
    # 找面积最大轮廓的外接矩形
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 255, 0), 1)
    if y > landmarks[0][1]:
        value = scale[0]
    elif y <= landmarks[len(landmarks) - 1][1]:
        value = scale[len(landmarks) - 1]
    else:
        for i in range(len(landmarks) - 1):
            if y <= landmarks[i][1] and y > landmarks[i + 1][1]:
                ratio = (landmarks[i][1] - y) / (landmarks[i][1] - landmarks[i + 1][1])
                value = scale[i] + round((scale[i + 1] - scale[i]) * ratio, 2)
    return value


def youwei_14_postprocess(name, img_seg, landmarks, img_show, is_upper=False):
    ratio_value = -255
    scale_thr = 80

    # 计算圆心关键点到每个表盘关键点所组成向量的角度
    angle_scale = []
    for i in range(len(landmarks) - 1):
        if landmarks[i][0] == landmarks[-1][0]:
            if landmarks[i][1] < landmarks[-1][1]:
                angle = 90
            else:
                angle = -90
        else:
            k = (landmarks[i][1] - landmarks[-1][1]) / (landmarks[i][0] - landmarks[-1][0])
            angle = math.atan(k)
            angle = int(angle * 180 / math.pi)
        # youwei_11的MIN角度0~90，MAX的角度-90~0, 由于拍摄角度倾斜会出现MIN小于0或者MAX大于0，需要修正角度
        if is_upper:
            if angle > 0 or i == 0:
                angle -= 90
            if angle < 0 or i == len(landmarks) - 2:
                angle += 90
        # youwei_14和youwei_15正常情况下MIN角度-90~0，MAX角度0~90，由于拍摄角度倾斜会出现MIN大于0或者MAX小于0，需要修正角度
        if i == 0 and angle > 0:
            angle = -180 + angle
        if i == len(landmarks) - 2 and angle < 0:
            angle = 180 + angle
        angle_scale.append(angle)
    # print(angle_scale, end=" ")

    # 判断估计的关键点坐标是不是顺时针的
    for i in range(len(angle_scale) - 1):
        if angle_scale[i] > angle_scale[i + 1]:
            return ratio_value

    # min和max夹角大于阈值scale_thr
    # print(angle_scale[-1] - angle_scale[0], end=" ")
    if angle_scale[-1] - angle_scale[0] < scale_thr:
        return ratio_value

    # 指针分割
    img_seg = np.array(img_seg[0], np.uint8)
    # 计算指针角度
    angle_point = pointer_angle(img_seg, img_show)

    if angle_point == -255:
        return ratio_value

    if is_upper:
        if angle_point > 0:
            angle_point -= 90
        else:
            angle_point += 90
    # print(angle_point, end=" ")

    # 计算指针与min夹角与max与min夹角的比值，归一化到0~1之间
    if angle_point > angle_scale[-1]:
        ratio_value = 1.00
    elif angle_point < angle_scale[0]:
        ratio_value = 0.00
    else:
        ratio_value = (angle_point - angle_scale[0]) / (angle_scale[-1] - angle_scale[0])
        ratio_value = round(ratio_value, 2)

    return ratio_value


def youwei_16_postprocess(name, img_seg, landmarks, img_show, is_upper=False, is_lower=False, is_rect=False, is_minrect=False):
    ratio_value = -255

    # 油位上限与下限中点
    tcp = (landmarks[1] + landmarks[-2]) / 2
    bcp = (landmarks[0] + landmarks[-1]) / 2
    if name == "youwei_13":
        tcp = (landmarks[8] + landmarks[9]) / 2
    cv2.circle(img_show, center=(int(tcp[0]), int(tcp[1])), radius=2, color=(255, 0, 0), thickness=-1)
    cv2.circle(img_show, center=(int(bcp[0]), int(bcp[1])), radius=2, color=(255, 0, 0), thickness=-1)

    img_seg = np.array(img_seg[0], np.uint8)
    # 查找分割轮廓
    contours = findsortContours(img_seg)
    ## cv2.drawContours(img_show, contours[0], -1, (255, 0, 0), 1)
    # 没有分割出指针区域
    if len(contours) == 0:
        return ratio_value
    cv2.drawContours(img_show, contours[0], -1, (255, 0, 0), 1)
    if is_rect:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 255, 0), 1)
        lpl = (x, y)
        lpr = (x + w, y)
    if is_minrect:
        # 油位最小矩形框
        rect, box = findminAreaRect(contours[0])
        cv2.drawContours(img_show, [box], 0, (0, 0, 255), 1)
        # 最小外接矩形上界或下界左右端点
        # 最小外接矩形上界左右端点
        if is_upper:
            if rect[0][0] < box[0][0]:
                lpl = box[2]
                lpr = box[3]
            else:
                lpl = box[1]
                lpr = box[2]
        if is_lower == True:
            # 最小外接矩形下界左右端点
            if rect[0][0] < box[0][0]:
                lpl = box[1]
                lpr = box[0]
            else:
                lpl = box[0]
                lpr = box[3]
    # 判断有油位线是否高于上下限
    k, b = LineEquation(lpl[0], lpl[1], lpr[0], lpr[1])
    # 油位线高于上限
    if (tcp[1] - k * tcp[0] - b) > 0:
        ratio_value = 1.0
    # 油位线低于下限
    elif (bcp[1] - k * bcp[0] - b) < 0:
        ratio_value = 0.0
    # 油位线处于上限与下限之间
    else:
        # 求油位上限与下限中点组成的直线与油位线的交点
        try:
            cross_point = findIntersection(tcp[0], tcp[1], bcp[0], bcp[1],
                                        lpl[0], lpl[1], lpr[0], lpr[1])
            cv2.circle(img_show, center=(int(cross_point[0]), int(cross_point[1])),
                    radius=2, color=(255, 0, 0), thickness=-1)
            # 油位
            ratio_value = (bcp[1] - cross_point[1]) / (bcp[1] - tcp[1])
        except:
            return ratio_value
    return ratio_value


def youwei_19_postprocess(img_seg, landmarks, scale, img_show):
    value = -255

    # 判断估计的关键点坐标是不是递增的
    scale_dist_list = []
    for i in range(len(landmarks) - 1):
        if landmarks[i][1] < landmarks[i + 1][1]:
            return value
        # 相邻刻度距离
        scale_dist_list.append(math.sqrt((landmarks[i + 1][0] - landmarks[i][0]) ** 2 + (landmarks[i + 1][1] - landmarks[i][1]) ** 2))

    # 相邻刻度距离均值
    scale_dist_mean = np.mean(scale_dist_list)
    # 相邻刻度距离中值
    scale_dist_median = np.median(scale_dist_list)
    # print(scale_dist_mean, scale_dist_median, end=" ")
    # 相邻刻度距离必须大于0.5倍中值且小于1.5倍中值
    for scale_dist in scale_dist_list:
        if scale_dist > 1.5*scale_dist_median or scale_dist < 0.5*scale_dist_median:
            # print(scale_dist, end=" ")
            return value

    img_seg = np.array(img_seg[0], np.uint8)
    # 油位框
    contours = findsortContours(img_seg)
    if len(contours) == 0:
        value = 0.0
        return value
    # 油位最小矩形框
    rect, box = findminAreaRect(contours[0])
    # cv2.drawContours(img_show, [box], 0, (0, 0, 255), 1)
    # 最小外接矩形上界左右端点
    if rect[0][0] < box[0][0]:
        rtpl = box[2]
        rtpr = box[3]
    else:
        rtpl = box[1]
        rtpr = box[2]
    cv2.line(img_show, (rtpl[0], rtpl[1]), (rtpr[0], rtpr[1]), (0, 255, 0), 1)
    # 判断有油位线是否高于上下限
    k, b = LineEquation(rtpl[0], rtpl[1], rtpr[0], rtpr[1])
    if (landmarks[0][1] - k * landmarks[0][0] - b) < 0:  # 油位线低于下限
        cross_point = findIntersection(rtpl[0], rtpl[1], rtpr[0], rtpr[1], landmarks[0][0], landmarks[0][1],landmarks[-1][0], landmarks[-1][1])
        cv2.circle(img_show, center=(int(cross_point[0]), int(cross_point[1])), radius=2, color=(255, 0, 0),thickness=-1)
        ratio = math.sqrt((landmarks[0][0] - cross_point[0]) ** 2 + (landmarks[0][1] - cross_point[1]) ** 2) / scale_dist_mean
        value = scale[1] - ratio*(scale[1]-scale[0])
        if value < scale[0]:
            value = scale[0]
        return value
    if (landmarks[-1][1] - k * landmarks[-1][1] - b) > 0:  # 油位线高于上限
        cross_point = findIntersection(rtpl[0], rtpl[1], rtpr[0], rtpr[1], landmarks[0][0], landmarks[0][1],landmarks[-1][0], landmarks[-1][1])
        cv2.circle(img_show, center=(int(cross_point[0]), int(cross_point[1])), radius=2, color=(255, 0, 0),thickness=-1)
        ratio = math.sqrt((landmarks[-1][0] - cross_point[0]) ** 2 + (landmarks[-1][1] - cross_point[1]) ** 2) / scale_dist_mean
        value = scale[-2] + ratio * (scale[-1] - scale[-2])
        if value>scale[-1]:
            value = scale[-1]
        return value
    # 油位在刻度之间
    for i in range(len(landmarks) - 1):
        if (landmarks[i][1] - k * landmarks[i][0] - b) * (landmarks[i+1][1] - k * landmarks[i+1][0] - b) <= 0:
            # 相交点
            cross_point = findIntersection(rtpl[0], rtpl[1], rtpr[0], rtpr[1], landmarks[i][0], landmarks[i][1], landmarks[i + 1][0], landmarks[i + 1][1])
            cv2.line(img_show, (landmarks[i][0], landmarks[i][1]), (landmarks[i + 1][0], landmarks[i + 1][1]), (0, 0, 255), 1)
            cv2.circle(img_show, center=(int(cross_point[0]), int(cross_point[1])), radius=2, color=(255, 0, 0), thickness=-1)
            ratio = math.sqrt((landmarks[i][0] - cross_point[0]) ** 2 + (landmarks[i][1] - cross_point[1]) ** 2) / math.sqrt((landmarks[i + 1][0] - landmarks[i][0]) ** 2 + (landmarks[i + 1][1] - landmarks[i][1]) ** 2)
            value = scale[i+1] + round((scale[i + 2] - scale[i+1]) * ratio, 2)
            return value


def youwei_20_postprocess(img_seg, landmarks, scale, img_show, has_center=True):
    value = -255

    # 判断是否估计圆心
    if has_center:
        scale_point = landmarks[:-1]
        circle_center = landmarks[-1]
    else:
        scale_point = landmarks
    # 指针分割
    img_seg = np.array(img_seg[0], np.uint8)
    # 查找指针分割轮廓
    contours = findsortContours(img_seg)
    ## cv2.drawContours(img_show, contours[0], -1, (255, 0, 0), 1)
    # 没有分割出指针区域
    if len(contours) == 0:
        return value
    cv2.drawContours(img_show, contours[0], -1, (255, 0, 0), 1)
    # 计算指针外接矩形框中心，以及圆心到形框中心的向量
    if has_center:
        # 找面积最小的矩形
        rect = cv2.minAreaRect(contours[0])
        cx, cy = int(rect[0][0]), int(rect[0][1])
        cv2.circle(img_show, center=(cx, cy), radius=2, color=(255, 0, 0), thickness=-1)
        cv2.line(img_show, (cx, cy), (circle_center[0], circle_center[1]), (0, 0, 255), 1)
        # 指针方向向量
        vector1 = [cx-circle_center[0], cy-circle_center[1]]
    # 指针直线拟合
    rows, cols = img_show.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(contours[0], cv2.DIST_L12, 0, 0.01, 0.01)
    # 控制指针与图像边框交点值的范围
    # 与图像高度h相交
    if math.fabs(vy/vx)<1:
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        point1 = [cols - 1, righty]
        point2 = [0, lefty]
    # 与图像宽度w相交
    else:
        upx = int((-y*vx/vy)+x)
        downx = int(((rows-y)*vx/vy)+x)
        point1 = [upx, 0]
        point2 = [downx, rows-1]
    cv2.line(img_show, (point1[0], point1[1]), (point2[0], point2[1]), (0, 255, 0), 1)
    # 指针直线方程一般式
    A, B, C = LineEquation(point1[0], point1[1], point2[0], point2[1], is_general=True)
    cross_point = []
    for i in range(len(scale_point)-1):
        # 指针直线方程在两相邻刻度之间
        if (A * scale_point[i][0] + B * scale_point[i][1] + C) * (A * scale_point[i + 1][0] + B * scale_point[i + 1][1] + C) <= 0:
            # 相交点
            cross_point = findIntersection(point1[0], point1[1], point2[0], point2[1], scale_point[i][0],
                                           scale_point[i][1], scale_point[i + 1][0], scale_point[i + 1][1])
            if has_center:
                # 从圆心到交点方向向量
                vector2 = [cross_point[0] - circle_center[0], cross_point[1] - circle_center[1]]
                # 圆心到交点方向向量与指针方向向量同向
                if (vector1[0] * vector2[0] + vector1[1] * vector2[1]) <= 0:
                    continue
            cv2.line(img_show, (scale_point[i][0], scale_point[i][1]),
                     (scale_point[i + 1][0], scale_point[i + 1][1]), (0, 0, 255), 1)
            cv2.circle(img_show, center=(int(cross_point[0]), int(cross_point[1])), radius=2, color=(255, 0, 0),
                       thickness=-1)
            ratio = math.sqrt(((scale_point[i][0] - cross_point[0]) ** 2 + (scale_point[i][1] - cross_point[1]) ** 2) /
                              ((scale_point[i + 1][0] - scale_point[i][0]) ** 2 + (scale_point[i + 1][1] - scale_point[i][1]) ** 2))
            value = scale[i] + round((scale[i + 1] - scale[i]) * ratio, 2)
            return value

    # if has_center:
    #     vector3 = (landmarks[0][0]-landmarks[-2][0], landmarks[0][1]-landmarks[-2][1])
    #     if (vector1[0]*vector3[0]+vector1[1]*vector3[1])>0:
    #         value = scale[0]
    #     else:
    #         value = scale[-1]
    #     return value

    # 指针小于最低或者大于最高刻度点情况判断属于最低还是最高
    cross_point = findIntersection(point1[0], point1[1], point2[0], point2[1], scale_point[0][0], scale_point[0][1],
                                   scale_point[-1][0], scale_point[-1][1])
    distance_start = math.sqrt((scale_point[0][0] - cross_point[0]) ** 2 + (scale_point[0][1] - cross_point[1]) ** 2)
    distance_end = math.sqrt((scale_point[-1][0] - cross_point[0]) ** 2 + (scale_point[-1][1] - cross_point[1]) ** 2)
    if distance_start <= distance_end:
        value = scale[0]
    else:
        value = scale[-1]
    return value


# 油位youwei_25后处理
def youwei_25_postprocess(name, img_seg, img_show, is_upper=False, is_lower=False):
    ratio_value = -255

    img_seg_line, img_seg_level = img_seg
    img_seg_line = np.array(img_seg_line, np.uint8)
    img_seg_level = np.array(img_seg_level, np.uint8)

    # 油位框
    contours = findsortContours(img_seg_line)
    if len(contours) == 0:
        return ratio_value
    # 油位最小矩形框
    rect, box = findminAreaRect(contours[0])
    # cv2.drawContours(img_show, [box], 0, (0, 0, 255), 1)

    if is_lower:
        # 最小外接矩形下界左右端点
        if rect[0][0] < box[0][0]:
            lpl = box[1]
            lpr = box[0]
        else:
            lpl = box[0]
            lpr = box[3]
    if is_upper:
        if rect[0][0] < box[0][0]:
            lpl = box[2]
            lpr = box[3]
        else:
            lpl = box[1]
            lpr = box[2]
    cv2.line(img_show, (lpl[0], lpl[1]), (lpr[0], lpr[1]), (0, 255, 0), 1)

    # 上下限最小矩形框
    contours = findsortContours(img_seg_level)
    if len(contours) < 4:
        return ratio_value
    lcp = []
    for i in range(4):
        rect, box = findminAreaRect(contours[i])
        lcp.append(rect[0])
        cv2.drawContours(img_show, [box], 0, (0, 255, 0), 1)
        cv2.circle(img_show, center=(int(rect[0][0]), int(rect[0][1])), radius=2,
                   color=(255, 0, 0), thickness=-1)
    lcp.sort(key=lambda x: x[1], reverse=True)
    tcp_x, tcp_y = (lcp[-1][0]+lcp[-2][0])/2, (lcp[-1][1]+lcp[-2][1])/2
    bcp_x, bcp_y = (lcp[0][0]+lcp[1][0])/2, (lcp[0][1]+lcp[1][1])/2
    cv2.line(img_show, (int(tcp_x), int(tcp_y)), (int(bcp_x), int(bcp_y)), (255, 0, 0), 1)

    cross_point = findIntersection(lpl[0], lpl[1], lpr[0], lpr[1], tcp_x, tcp_y, bcp_x, bcp_y)
    cv2.circle(img_show, center=(int(cross_point[0]), int(cross_point[1])),
               radius=2, color=(0, 0, 255), thickness=-1)
    # 油位线高于上限
    if cross_point[1] < tcp_y:
        ratio_value = 1.0
    # 油位线低于下限
    elif cross_point[1] > bcp_y:
        ratio_value = 0.0
    else:
        ratio_value = (bcp_y - cross_point[1]) / (bcp_y - tcp_y)

    return ratio_value


# 油位youwei_18后处理
def youwei_18_postprocess(img_seg, img_show):
    value = -255

    img_seg_line, img_seg_level = img_seg
    img_seg_line = np.array(img_seg_line, np.uint8)
    img_seg_level = np.array(img_seg_level, np.uint8)

    # 油位框
    contours = findsortContours(img_seg_line)
    if len(contours)==0:
        return value
    # 油位最小矩形框
    rect, box = findminAreaRect(contours[0])
    cv2.drawContours(img_show, [box], 0, (0, 0, 255), 1)

    # 最小外接矩形下界最低点
    bottom = box[0]
    cv2.circle(img_show, center=(int(bottom[0]), int(bottom[1])), radius=2,
               color=(255, 0, 0), thickness=-1)

    # 上下限最小矩形框
    contours = findsortContours(img_seg_level)
    if len(contours) < 2:
        return value
    lcp = []
    for i in range(2):
        rect, box = findminAreaRect(contours[i])
        lcp.append(rect[0])
        cv2.drawContours(img_show, [box], 0, (0, 255, 0), 1)
        cv2.circle(img_show, center=(int(rect[0][0]), int(rect[0][1])), radius=2,
                   color=(255, 0, 0), thickness=-1)

    # 判断油位线是否高于下限
    k, b = LineEquation(lcp[0][0], lcp[0][1], lcp[1][0], lcp[1][1])
    # 油位线低于下限
    if (bottom[1] - k * bottom[0] - b) > 0:
        value = 0.0
    else:
        value = 1.0
    return value


# 直线方程
def LineEquation(first_x, first_y, second_x, second_y, is_general=False):
    # 一般式 Ax+By+C=0
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    if is_general:
        return A, B, C
    k = -1 * A / B
    b = -1 * C / B
    return k, b


def pointer_angle(img_seg, img_show):
    angle = -255
    # 查找指针分割轮廓
    contours = findsortContours(img_seg)
    ## cv2.drawContours(img_show, contours[0], -1, (255, 0, 0), 1)
    # 没有分割出指针区域
    if len(contours) == 0:
        return angle
    cv2.drawContours(img_show, contours[0], -1, (255, 0, 0), 1)

    [vx, vy, x, y] = cv2.fitLine(contours[0], cv2.DIST_L12, 0, 0.01, 0.01)
    angle = math.atan(vy/vx)
    angle = int(angle * 180 / math.pi)
    return angle


# 直线交点
def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return [px, py]


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


# 查找轮廓最小外接矩形
def findminAreaRect(contour):
    # 找面积最小的矩形
    rect = cv2.minAreaRect(contour)
    # 得到最小矩形的坐标
    box = cv2.boxPoints(rect)
    # 标准化坐标到整数
    box = np.int0(box)
    return rect, box