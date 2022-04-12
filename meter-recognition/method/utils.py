# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import cv2
from scipy import optimize
import configure
from sklearn.decomposition import PCA
import json


# 求解线段和射线的交点
class dotPair():  # 点对线段
    def __init__(self, sPoint=[0, 0], ePoint=[1, 1]):
        self.sPoint = sPoint
        self.ePoint = ePoint


# 对于光线来说，由于只有一个端点，所以必须再有一个角度以确定其方向。
class line():
    def __init__(self, sPoint=[0, 0], ePoint=[1, 1]):  # img_shape=[0,0]
        # self.sPoint=[sPoint[0],img_shape[0]-sPoint[1]]
        # self.ePoint=[ePoint[0],img_shape[0]-ePoint[1]]
        self.sPoint = sPoint
        self.ePoint = ePoint
        bevel = math.sqrt(math.pow(self.ePoint[1] - self.sPoint[1], 2) + math.pow(self.ePoint[0] - self.sPoint[0], 2))
        cos_theta = (ePoint[0] - sPoint[0]) / bevel
        sin_theta = (ePoint[1] - sPoint[1]) / bevel  # 符号是因为图像中坐标y轴和笛卡尔坐标y轴反方向；
        self.theta = [cos_theta, sin_theta]
        self.abc = np.array([self.theta[1], -(self.theta[0]), -(np.linalg.det([self.sPoint, self.theta]))],
                            dtype='float')
        # 其中，abc表示该射线的方程，theta为角度，用三角函数表示; sPoint表示射线的端点。


class Points2Circle(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_m = np.mean(x)
        self.y_m = np.mean(y)

    def calc_r(self, xc, yc):
        return np.sqrt((self.x - xc)**2 + (self.y - yc)**2)

    def fun(self, c):
        ri = self.calc_r(*c)
        return ri - ri.mean()

    def process(self):
        center_estimate = self.x_m, self.y_m
        center = optimize.leastsq(self.fun, center_estimate)
        center = center[0]
        r = self.calc_r(*center)
        r = r.mean()
        return center, r


def read_json(json_path):
    f = open(json_path, encoding='utf-8')
    json_info = json.load(f)
    shapes = json_info['shapes']

    points = []
    for i in range(len(shapes)):
        point = shapes[i]['points'][0]
        points.append(point)

    points = np.asarray(points, dtype=np.float32)

    f.close()
    return points


def pca_radial_ellipse(img, center):
    # 利用pca进行直线检测
    flag = 1
    point1_x = 0
    point1_y = 0
    point2_x = 0
    point2_y = 0
    # img = np.zeros((10,10))
    # img_pointer = np.where(img==255)
    img_pointer = np.where(img != 0)
    if len(img_pointer[0]) == 0:
        flag = 0
        return flag, (int(point1_x), int(point1_y)), (int(point2_x), int(point2_y))
    img_pointer_np = np.array(img_pointer)
    img_pointer_np = img_pointer_np.T
    pca = PCA(n_components=1)
    # pca = cv2.PCACompute()
    new = pca.fit_transform(img_pointer_np)
    x_sum = 0
    y_sum = 0
    # 指针中心像素坐标（x_mean,y_sum）
    for i in range(img_pointer_np.shape[0]):
        y = img_pointer_np[i][0]
        x = img_pointer_np[i][1]
        x_sum = x + x_sum
        y_sum = y + y_sum
    x_mean = x_sum / img_pointer_np.shape[0]
    y_mean = y_sum / img_pointer_np.shape[0]
    # 指针所在直线上的两点（point1_x,point1_y）,（point2_x,point2_y）
    if math.fabs(pca.components_[0][1]) == 0:  # 指针竖直
        point1_x = x_mean
        point2_x = x_mean
        point1_y = y_mean
        if y_mean > center[1]:
            point2_y = img.shape[0]
        else:
            point2_y = 0
    if math.fabs(pca.components_[0][0]) == 0:  # 指针水平
        point1_y = y_mean
        point2_y = y_mean
        point1_x = x_mean
        if x_mean > center[0]:
            point2_x = img.shape[1]
        else:
            point2_x = 0
    if math.fabs(pca.components_[0][0]) != 0 and math.fabs(pca.components_[0][1]) != 0:
        # 指针方向判别
        point1_x = x_mean
        point1_y = y_mean
        b = y_mean - (pca.components_[0][0] / pca.components_[0][1]) * x_mean
        k = pca.components_[0][0] / pca.components_[0][1]
        if math.fabs(x_mean - center[0]) >= math.fabs(y_mean - center[1]):
            if x_mean >= center[0]:
                point2_x = img.shape[1]
                point2_y = k * point2_x + b
            else:
                point2_x = 0
                point2_y = k * point2_x + b
        else:
            if y_mean >= center[1]:
                point2_y = img.shape[0]
                point2_x = (point2_y - b) / k
            else:
                point2_y = 0
                point2_x = (point2_y - b) / k
    return flag, (int(point1_x), int(point1_y)), (int(point2_x), int(point2_y))


# 判定是否相交
def ifIntersect(line1=line(), line2=dotPair()):  # img_shape=[0,0]
    sFlag = np.sum(np.array(line1.abc) * np.array([line2.sPoint[0], line2.sPoint[1], 1]))
    eFlag = np.sum(np.array(line1.abc) * np.array([line2.ePoint[0], line2.ePoint[1], 1]))
    num = 0
    num += sFlag == 0
    num += eFlag == 0
    num += sFlag * eFlag < 0
    return num
    # 其返回值表示射线与线段的交点个数，当交点个数为2时，表示线段在射线上。


# 求取交点
# 为了得到交点，则必须得到点对的abc参数。然后调用numpy.linalg中的solve函数求解由两条直线的系数组成的方程组，进而得到二者交点。
def cross(l1=line(), l2=dotPair()):
    flag_cross = 0
    num = ifIntersect(l1, l2)
    if num == 2 or num == 0:
        # print('线段在射线上或没有交点')
        pt = (-255, -255)
        return pt, flag_cross
    else:
        abLine = [l1.abc[0], l1.abc[1]]
        abDot = [l2.ePoint[1] - l2.sPoint[1], l2.sPoint[0] - l2.ePoint[0]]
        c = [[-l1.abc[2]], [-(l2.ePoint[0] * l2.sPoint[1] - l2.ePoint[1] * l2.sPoint[0])]]
        pt = np.linalg.solve([abLine, abDot], c)
        if math.fabs(pt[0]) == 0:
            pt[0] = 0
        if math.fabs(pt[1]) == 0:
            pt[1] = 0
        if (pt[0][0] - l1.sPoint[0]) * l1.theta[0] >= 0 and (
                pt[1][0] - l1.sPoint[1]) * l1.theta[1] >= 0:  # 判断交点是否在射线上（在射线延长线上不算是在射线上）
            flag_cross = 1
        return pt, flag_cross


def cross_blq(l1=line(), l2=dotPair()):
    flag_cross = 0
    num = ifIntersect(l1, l2)
    # if num==1: # 线段与射线存在一个交点
    abLine = [l1.abc[0], l1.abc[1]]
    abDot = [l2.ePoint[1] - l2.sPoint[1], l2.sPoint[0] - l2.ePoint[0]]
    c = [[-l1.abc[2]], [-(l2.ePoint[0] * l2.sPoint[1] - l2.ePoint[1] * l2.sPoint[0])]]
    try:
        pt = np.linalg.solve([abLine, abDot], c)
    except (Exception):
        pt = [[-255], [-255]]
        # print('交点定位错误！')
    pt = [pt[0][0], pt[1][0]]
    if math.fabs(pt[0]) == 0:
        pt[0] = 0
    if math.fabs(pt[1]) == 0:
        pt[1] = 0
    if (pt[0] - l1.sPoint[0]) * l1.theta[0] >= 0 and (pt[1] - l1.sPoint[1]) * l1.theta[1] >= 0:  # 判断交点是否在射线上；
        flag_cross = 1
    return pt, flag_cross


def get_pointer(filename, img_adjust, img_seg_new, center, r_out, r_in):
    # 根据关键点位置和表盘表针分割结果，获取有效指针前景
    circle = np.zeros((img_adjust.shape[0], img_adjust.shape[1]), dtype="uint8") * 255
    cv2.circle(circle, (int(center[0]), int(center[1])), r_out, 255, -1)
    cv2.circle(circle, (int(center[0]), int(center[1])), r_in, 0, -1)
    # cv2.imwrite(os.path.join(configure.config.save_dir,filename[:-4]+'circle.jpg'),circle)
    nrows, ncols = circle.shape[:2]
    row, col = np.ogrid[:nrows, :ncols]
    if 'JCQ' in configure.config.name:
        circle[np.where(row > int(center[1])), :] = 0  # 针对避雷针表计，只取圆形区域的上半圆，以防分割结果中的噪音干扰。
    # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4] + '_circle.jpg'), circle)
    img_pointer = np.zeros_like(circle)
    img_pointer[circle != 0] = img_seg_new[circle != 0]
    # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4]+'circle_pointer.jpg'),img_pointer)
    return img_pointer


def distance(x, y):
    dis = math.sqrt(math.pow(x[1] - y[1], 2) + math.pow(x[0] - y[0], 2))
    return dis


def scale_mean(landmarks):
    points_key = list(landmarks)
    point_distance_sum = 0
    for num_point in range(len(points_key) - 1):
        # if num_point%2 !=0:
        #     continue
        point_distance = distance(points_key[num_point + 1], points_key[num_point])
        point_distance_sum = point_distance_sum + point_distance
    point_distance_mean = point_distance_sum / ((len(points_key) - 1))
    return point_distance_mean


def flagS(pt, p, f):
    dis_flag = -1
    if math.fabs(pt[0] - p[0]) > math.fabs(pt[1] - p[1]):  # 交点与0点之间，横坐标距离大于纵坐标距离
        if (pt[0] - p[0] <= 0 and f == 1) or (pt[0] - p[0] >= 0 and (f == 2 or f == 3)):  #
            dis_flag = 1
        else:
            dis_flag = 0
    else:
        if (pt[1] - p[1] >= 0 and (f == 1 or f == 2)) or (pt[1] - p[1] <= 0 and f == 3):
            dis_flag = 1
        else:
            dis_flag = 0
    return dis_flag


def flagE(pt, p, f):
    dis_flag = -1
    if math.fabs(pt[0] - p[0]) > math.fabs(pt[1] - p[1]):  # 交点与0点之间，横坐标距离大于纵坐标距离
        if (pt[0] - p[0] >= 0 and (f == 1 or f == 3)) or (pt[0] - p[0] < 0 and f == 2):  #
            dis_flag = 1
        else:
            dis_flag = 0
    else:
        if pt[1] - p[1] >= 0:
            dis_flag = 1
        else:
            dis_flag = 0
    return dis_flag


def recognition_ellipse(filename, scale_carve, img_show, scale_m, point1_x, point1_y, point2_x, point2_y, landmarks):
    pointer = -255
    # 相邻主刻度（feature_point）读数差
    # scale = scale_carve[1]-scale_carve[0]
    # 求解主刻度连线与指针所在直线的交点
    for i in range(landmarks.shape[0] - 1):
        # 刻度线 所在直线上的两个点的坐标scale_point_0 scale_point_1
        scale_point_0 = landmarks[i]
        scale_point_1 = landmarks[i + 1]
        try:
            cv2.line(img_show, (scale_point_0[0], scale_point_0[1]), (scale_point_1[0], scale_point_1[1]), (0, 255, 0), 1)
        except:
            pass
        scale = scale_carve[i + 1] - scale_carve[i]
        # 相邻刻度点
        Dot = dotPair(sPoint=scale_point_0, ePoint=scale_point_1)
        Line = line(sPoint=[point1_x, point1_y], ePoint=[point2_x, point2_y])
        pt, flag_cross = cross_blq(Line, Dot)  # pt:交点  flag_cross：标志是否有交点；如果为0，则表示无交点；如果为1，表示1个交点；如果为2，表示线段在射线上。
        try:
            cv2.circle(img_show, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
        except:
            print(int(pt[0]), int(pt[1]))
        #cv2.circle(img_show, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
        cv2.line(img_show, (point1_x, point1_y), (point2_x, point2_y), (0, 255, 0), 1)
        # cv2.line(img_show, (int(scale_point_0[0]), int(scale_point_0[1])),
        #          (int(scale_point_1[0]), int(scale_point_1[1])), (0, 255, 0), 1)
        # cv2.line(img_show,(int(scale_point_0[0]),int(scale_point_0[1])),(int(scale_point_1[0]),int(scale_point_1[1])),(0,255,0),1)
        # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4]+'_pt.jpg'),img_show)

        if i == 0 and flagS(pt, landmarks[i], configure.config.flag) == 1 and flag_cross != 0:
            # if ((pt[0] - landmarks[0][0] >= 0 or (pt[1] - landmarks[0][1] <= 0))  and configure.config.flag == 1 ) or
            # ((pt[0] - landmarks[0][0]<=0 or pt[1] - landmarks[0][1] >= 0 ) and configure.config.flag == 0 ):  # 指针指向0刻度以下
            if pt[0] == -255 and pt[1] == -255:
                print('交点定位错误！')
            scale_distance = math.sqrt(
                math.pow(scale_point_1[1] - scale_point_0[1], 2) + math.pow(scale_point_1[0] - scale_point_0[0], 2))
            # if math.fabs(scale_distance - scale_m) > scale_m / 3:  # 当两点之间的距离与均值距离的差值大于1/3的标准距离时，认为特征点定位错误；
            #     print('识别置信度未达标，请进行人工校验！')
            # cv2.circle(img_pt, center=(int(pt[0]), int(pt[1])), radius=2, color=(0, 255, 0), thickness=-1)
            # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4] + '_pt.jpg'), img_pt)
            pointer = scale_carve[0]
            # print('pointer={}'.format(str(pointer)[:5]))
            cv2.circle(img_show, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), -1)
            cv2.line(img_show, (point1_x, point1_y), (int(pt[0]), int(pt[1])), (0, 255, 0), 1)
            cv2.putText(img_show, str(scale_carve[0])[:5], (50, 240), cv2.FONT_ITALIC, 2.0, (0, 255, 0), 3)
            # img_show = cv2.resize(img_show, (640,640), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(configure.config.save_dir, filename), img_show)
            return pointer
        if i == landmarks.shape[0] - 2 and flagE(pt, landmarks[-1], configure.config.flag) == 1 and flag_cross != 0:
            if pt[0] == -255 and pt[1] == -255:
                print('交点定位错误！')
            scale_distance = math.sqrt(
                math.pow(scale_point_1[1] - scale_point_0[1], 2) + math.pow(scale_point_1[0] - scale_point_0[0], 2))
            # if math.fabs(scale_distance - scale_m) > scale_m / 3:
            #     print('识别置信度未达标，请进行人工校验！')
            pointer = scale_carve[-1]
            # print('pointer={}'.format(str(pointer)[:5]))
            # cv2.circle(img_pt, center=(int(pt[0]), int(pt[1])), radius=2, color=(0, 255, 0), thickness=-1)
            # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4] + '_pt.jpg'), img_pt)
            cv2.circle(img_show, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), -1)
            cv2.line(img_show, (point1_x, point1_y), (int(pt[0]), int(pt[1])), (0, 255, 0), 1)
            cv2.putText(img_show, str(scale_carve[-1])[:5], (50, 240), cv2.FONT_ITALIC, 2.0, (0, 255, 0), 3)
            # img_show = cv2.resize(img_show, (640,640), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(configure.config.save_dir, filename), img_show)
            return pointer
        # if i == 0 and pt[0] - landmarks[0][0] <= 0: # configure.config.pointer_class=='56'
        if flag_cross != 0 and (pt[0] - scale_point_0[0]) * (pt[0] - scale_point_1[0]) <= 0 and (
                pt[1] - scale_point_0[1]) * (pt[1] - scale_point_1[1]) <= 0:
            if pt[0] == -255 and pt[1] == -255:
                print('交点定位错误！')
            scale_distance = math.sqrt(
                math.pow(scale_point_1[1] - scale_point_0[1], 2) + math.pow(scale_point_1[0] - scale_point_0[0], 2))
            # if configure.config.name == 'youwen_1' and i == 4:
            #     if math.fabs(scale_distance - scale_m) > (scale_m + scale_m / 3):
            #         print('识别置信度未达标，请进行人工校验！')
            # else:
            #     if math.fabs(scale_distance - scale_m) > scale_m / 3:
            #         print('识别置信度未达标，请进行人工校验！')
            # 交点pt与左刻度点的距离
            pt_x_distance = pt[0] - scale_point_0[0]
            pt_y_distance = pt[1] - scale_point_0[1]
            pt_distance = math.sqrt(pt_x_distance**2 + pt_y_distance**2)
            # 右刻度点与左刻度点的距离
            scale_x_distance = scale_point_1[0] - scale_point_0[0]
            scale_y_distance = scale_point_1[1] - scale_point_0[1]
            scale_distance = math.sqrt(scale_x_distance**2 + scale_y_distance**2)
            # pointer = scale_carve[int(i/2)] + configure.config.scale_55 * pt_distance / scale_distance
            pointer = scale_carve[i] + scale * pt_distance / scale_distance
            # if configure.config.name=='dangwei_1':
            #     pointer = math.floor(pointer)
            # cv2.circle(img_pt, center=(int(pt[0]), int(pt[1])), radius=2, color=(0, 255, 0), thickness=-1)
            # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4] + '_pt.jpg'), img_pt)
            # print('pointer={}'.format(str(pointer)[:5]))
            cv2.circle(img_show, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), -1)
            cv2.line(img_show, (point1_x, point1_y), (int(pt[0]), int(pt[1])), (0, 255, 0), 1)
            cv2.putText(img_show, str(pointer)[:5], (50, 240), cv2.FONT_ITALIC, 2.0, (0, 255, 0), 3)
            # img_show = cv2.resize(img_show, (640,640), interpolation=cv2.INTER_CUBIC)
            # img_save = cv.resize(img, (resize_w, resize_h), interpolation=cv.INTER_CUBIC)
            cv2.imwrite(os.path.join(configure.config.save_dir, filename), img_show)
            return pointer

    # 指针定位错误
    if pointer == -255.0:
        cv2.putText(img_show, 'error!', (50, 240), cv2.FONT_ITALIC, 2.0, (0, 255, 0), 3)
        cv2.imwrite(os.path.join(configure.config.save_dir, filename), img_show)
        return pointer


def segment_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 检测黑针
    if configure.config.name == 'dangwei_1':
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 55])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        # black_mask[black_mask<128]=0
        # black_mask[black_mask>127]=255
        return black_mask
    if configure.config.name == 'youwei_1':
        # 检测红针
        lower_red_1 = np.array([156, 43, 46])
        upper_red_1 = np.array([180, 255, 255])

        lower_red_2 = np.array([0, 43, 46])
        upper_red_2 = np.array([10, 255, 255])

        red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        red_mask = red_mask_1 + red_mask_2
        return red_mask
    # 如果不是上述表计，则返回全零矩阵；
    error = np.zeros(img.shape[0], img.shape[1])
    return error


def pca_angle(img):
    # 利用 pca 进行直线角度检测
    angle = 0
    flag = 1
    img_pointer = np.where(img != 0)
    if len(img_pointer[0]) == 0:
        flag = 0
        return flag, angle
    img_pointer_np = np.array(img_pointer)
    img_pointer_np = img_pointer_np.T
    pca = PCA(n_components=1)
    pca.fit_transform(img_pointer_np)
    k = pca.components_[0][0] / pca.components_[0][1]
    angle = math.atan(k)
    angle = int(angle * 180 / math.pi)
    return flag, angle


# 获取最大连通域
def connect_demain(img):
    # w, h, n = img.shape
    # im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    flag = 1
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # cv2.imshow('th',thresh)
    # nccomps = cv2.connectedComponentsWithStats(thresh)#labels,stats,centroids
    nlabel, labels, status, centroids = cv2.connectedComponentsWithStats(thresh)  # labels,stats,centroids
    # max_connect = np.asarray(labels, 'uint8')
    max_connect = np.zeros_like(labels, 'uint8')
    if nlabel == 1:
        flag = 0
        # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4] + '_max_connect.jpg'), img)
        return img, flag
    # 去除背景
    background_idex = []
    for row in range(status.shape[0]):
        if status[row, :][0] == 0 and status[row, :][1] == 0:
            # background = row
            background_idex.append(row)
            # status_no_background = np.delete(status, background, axis=0)
        # else:
        #     continue
    status_no_background = status
    for idx, back_idx in enumerate(background_idex):
        status_no_background = np.delete(status_no_background, back_idx - idx, axis=0)
    # 提取最大连通域
    # idx = status_no_background[:, 4].argmax()
    re_value_max_position = np.asarray(status_no_background[:, 4].argmax())
    # h = np.asarray(labels, 'uint8')
    max_connect[labels == (re_value_max_position + len(background_idex))] = 255
    # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4]+'_max_connect.jpg'), max_connect)
    return max_connect, flag


def improve_landmark_with_template(landmarks):
    template_points = read_json(configure.config.temp_path)
    H, _ = cv2.findHomography(template_points, landmarks, cv2.RANSAC, 10)
    tmp_template_points = np.float32(template_points.reshape(-1, 1, 2))
    improved_landmark = cv2.perspectiveTransform(tmp_template_points, H).reshape(-1, 2)
    return improved_landmark
