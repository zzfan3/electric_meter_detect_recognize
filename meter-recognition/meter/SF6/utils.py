import os
import cv2
import math
import numpy as np
import configure
from method.utils import Points2Circle, read_json


def process_no_segreg(new_img, landmarks, filename):
    pointer = -255
    template_points = read_json(configure.config.temp_path)
    mat, mask = cv2.findHomography(landmarks, template_points, cv2.RANSAC, 10)
    if mat is None:
        # cv2.putText(res, str(pointer)[:5], (50, 240), cv2.FONT_ITALIC, 2.0, (0, 255, 0), 3)
        # print('pointer:{}'.format(str(pointer)[:6]))
        return pointer
    res = cv2.warpPerspective(new_img, mat, (256, 256))
    # res_show = np.copy(res)
    # cv2.imwrite(os.path.join(configure.config.save_dir,filename[:-4]+'_adjust.jpg'),res_show)
    # for i in range(template_points.shape[0]):
    #     x = int(template_points[i, 0])
    #     y = int(template_points[i, 1])
    #     cv2.circle(res, (x, y), 2, (255, 0, 0), 2)

    work = Points2Circle(template_points[:, 0], template_points[:, 1])
    center, r = work.process()

    # cv2.circle(res, (int(center[0]), int(center[1])), r,(255, 0, 0), 2)
    # cv2.imshow('res_show', res_show)
    # cv2.waitKey(0)
    center_x = int(center[0] + 0.5)
    center_y = int(center[1] + 0.5)
    center = (center_x, center_y)
    radius = int(r)

    r_in = radius - configure.config.radius_in_b  # r1 in
    r_out = radius + configure.config.radius_out_b  # r2 out
    crop_img_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    circle = np.zeros(crop_img_gray.shape[0:2], dtype="uint8")
    cv2.circle(circle, (center_x, center_y), r_out, 255, -1)
    cv2.circle(circle, (center_x, center_y), r_in, 0, -1)
    mask_img = cv2.bitwise_and(crop_img_gray, circle)
    # cv2.imshow('edges', mask_img)
    # cv2.waitKey(0)

    maxRadius = math.hypot(center_x, center_y)
    linear_polar = cv2.linearPolar(mask_img, (center_x, center_y), maxRadius, cv2.INTER_LINEAR)
    h, w, c = res.shape
    tmp_r1 = r_in * w / maxRadius
    tmp_r2 = r_out * w / maxRadius
    x1 = np.array([[tmp_r1, tmp_r2]], np.float32)
    y1 = np.array([[0, 0]], np.float32)
    r3, theta3 = cv2.cartToPolar(x1, y1, angleInDegrees=True)
    cumsum = np.sum(linear_polar[:, int(r3[0, 0]):int(r3[0, 1])], axis=1)
    cumsum = np.gradient(cumsum) * 2 + cumsum
    idx = np.argsort(cumsum)
    # cv2.line(linear_polar, (r3[0, 0], idx[0]), (r3[0, 1], idx[0]),
    #          (0, 0, 0), 2)
    # cv2.imshow("linear_polar", linear_polar)
    # cv2.waitKey(0)

    x1 = np.array([[r_in, r_out]], np.float32)
    y1 = np.array([[0, 0]], np.float32)
    r3, theta3 = cv2.cartToPolar(x1, y1, angleInDegrees=True)
    r = np.array([r3[0, 0], r3[0, 1]], np.float32)
    theta = np.array([idx[0], idx[0]], np.float32)
    theta = theta * 360 / h
    theta_fpoint = math.atan(
        (center[1] - template_points[0][1]) / (center[0] - template_points[0][0])) / math.pi * 180 + 180
    theta_lpoint = math.atan(
        (center[1] - template_points[-1][1]) / (center[0] - template_points[-1][0])) / math.pi * 180
    theta_scope = 360 - theta_fpoint + theta_lpoint
    if theta[0] <= theta_lpoint:
        pointer = configure.config.point_level + theta[0] * 1 / theta_scope
    else:
        if theta[0] >= theta_fpoint - 1:
            pointer = configure.config.scale_carve[0] + (theta[0] - theta_fpoint) * (configure.config.scale_carve[-1] -
                                                                           configure.config.scale_carve[0]) / theta_scope
    # cv2.putText(res, str(pointer)[:5], (50, 240), cv2.FONT_ITALIC, 2.0, (0, 255, 0), 3)
    # print('pointer:{}'.format(str(pointer)[:6]))
    #     # print('pointer={}'.format(pointer))
    x, y = cv2.polarToCart(r, theta, angleInDegrees=True)
    x1 = int(x[0, 0]) + center_x
    y1 = int(y[0, 0]) + center_y
    x2 = int(x[1, 0]) + center_x
    y2 = int(y[1, 0]) + center_y
    cv2.line(res, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # final_theta = theta[0]
    # cv2.imshow("res", res)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(configure.config.save_dir, filename), res)
    return pointer
