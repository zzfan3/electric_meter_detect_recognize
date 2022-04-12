#coding=utf-8
import cv2
import numpy as np
import configure
from third_part.hourglass.api import KeyPoint
from third_part.leinao_segmentation.api import LeiNaoUNet
from method.utils import Points2Circle, scale_mean, get_pointer, pca_radial_ellipse, recognition_ellipse
from method.utils import improve_landmark_with_template


class MethodJCQ(object):
    def __init__(self):
        super(MethodJCQ, self).__init__()
        self.kp = KeyPoint()
        self.sg = LeiNaoUNet(pointer_num=configure.config.pointer_num + 1)

    def __call__(self, img, filename, kp, seg):
        if kp is not None:
            landmarks = kp
            img_adjust = img
        else:
            landmarks, img_adjust = self.kp(img, configure.config.adjust_resolution, 128, configure.config.mean_std)
            landmarks = landmarks[:len(configure.config.scale_carve)]
            if configure.config.temp_path != 'None':
                landmarks = improve_landmark_with_template(landmarks)

        h_adj, w_adj, _ = img_adjust.shape
        img_show = img_adjust.copy()
        for point in landmarks:
            cv2.circle(img_show, center=(int(point[0]), int(point[1])), radius=2, color=(0, 0, 255), thickness=-1)
        # cv2.imwrite(os.path.join(save_dir,filename[:-4] + '_point.jpg'), img_show)
        img_showb = img_show.copy()
        # 获取表计关键点之间的平均距离
        scale_m = scale_mean(landmarks)
        # 计算特征点所在的圆心和半径
        work = Points2Circle(landmarks[:, 0], landmarks[:, 1])
        center, r = work.process()
        rb_out, rb_in = int(r) - configure.config.radius_out_b, int(r) - configure.config.radius_in_b
        if configure.config.radius_in_b == -255 or rb_in <= 0:
            rb_in = 0
        if rb_out <= 0:
            rb_out = 10

        if seg is not None:
            img_segb = seg[0]
        else:
            img_segb, _ = self.sg(img, configure.config.seg_resolution, 0, configure.config.adjust_resolution)[0]  # LeiNaoUnet分割结果
        # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4] + '_seg.jpg'), img_segb)
        # 获取指针区域
        pointer_imgb = get_pointer(filename, img_adjust, img_segb, center, rb_out, rb_in)
        # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4] + '_img_pointb.jpg'), pointer_imgb)

        if cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours(pointer_imgb.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        elif cv2.__version__.startswith('3'):
            _, contours, _ = cv2.findContours(pointer_imgb.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            raise AssertionError('cv2 must be either version 3 or 4 to call this method')
        max_h = 0
        max_h_idx = 0
        for contour_idx, contour in enumerate(contours):
            _, _, _, pointer_h = cv2.boundingRect(contour)
            if pointer_h > max_h:
                max_h = pointer_h
                max_h_idx = contour_idx
        new_pointer_imgb = np.zeros(shape=pointer_imgb.shape,
                                    dtype=pointer_imgb.dtype)
        cv2.drawContours(new_pointer_imgb, contours, max_h_idx, 255, -1)

        flagb, (point1_xb, point1_yb), (point2_xb, point2_yb) = pca_radial_ellipse(new_pointer_imgb, center)
        cv2.line(new_pointer_imgb, (point1_xb, point1_yb), (point2_xb, point2_yb), 255, 1)

        result_list = []
        if flagb == 0:
            print('指针定位失败，请进行人工识别！')
            result_list.append(-255.0)
        else:
            cv2.line(img_showb, (int(point1_xb), int(point1_yb)), (int(point2_xb), int(point2_yb)),
                     (0, 255, 0), 1)
            result_list.append(recognition_ellipse(filename, configure.config.scale_carve, img_showb, scale_m, point1_xb, point1_yb,
                               point2_xb, point2_yb, landmarks))
        return result_list


class JCQ(object):
    def __init__(self):
        super(JCQ, self).__init__()
        self.method = MethodJCQ()

    def __call__(self, img, filename, kp, seg):
        return self.method(img, filename, kp, seg)
