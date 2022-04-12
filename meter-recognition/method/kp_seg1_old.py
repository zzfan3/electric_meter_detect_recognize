import cv2
import configure
from third_part.hourglass.api import KeyPoint
from third_part.unet.api import OSegmentor
from .utils import scale_mean, Points2Circle, get_pointer, pca_radial_ellipse, recognition_ellipse, connect_demain
from .utils import improve_landmark_with_template


# 关键点+单指针分割+椭圆拟合
class MethodKpSegOneOld(object):
    def __init__(self):
        self.kp = KeyPoint()
        self.sg = OSegmentor()

    def __call__(self, img, filename, kp, seg):
        if kp is not None:
            landmarks = kp
            img_adjust = img
        else:
            landmarks, img_adjust = self.kp(img, configure.config.adjust_resolution, 128, configure.config.mean_std)
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
            img_segb, _ = self.sg(img, configure.config.seg_resolution, 0, configure.config.adjust_resolution)  # Unet分割结果
        # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4] + '_seg.jpg'), img_segb)
        # 获取指针区域
        pointer_imgb = get_pointer(filename, img_adjust, img_segb, center, rb_out, rb_in)
        # cv2.imwrite(os.path.join(configure.config.save_dir, filename[:-4] + '_img_pointb.jpg'), pointer_imgb)
        ellipse = cv2.fitEllipse(landmarks)
        # 获取最大联通区域
        pointer_imgb_max, flag_connect = connect_demain(pointer_imgb)
        # cv2.imwrite(os.path.join(save_dir, filename[:-4] + '_connect.jpg'), pointer_imgb_max)
        flagb, (point1_xb, point1_yb), (point2_xb,
                                        point2_yb) = pca_radial_ellipse(pointer_imgb_max, ellipse[0])
        cv2.line(pointer_imgb, (point1_xb, point1_yb), (point2_xb, point2_yb), 255, 1)

        result_list = []
        if flagb == 0:
            print('指针定位失败，请进行人工识别！')
            result_list.append(-255.0)
        else:
            result_list.append(recognition_ellipse(filename[:-4] + '_b.jpg', configure.config.scale_carve, img_showb, scale_m,
                                                   point1_xb, point1_yb, point2_xb, point2_yb, landmarks))
        return result_list