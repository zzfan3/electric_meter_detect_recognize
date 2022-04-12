import configure

from method.kp_seg1_new import MethodKpSegOneNew
from method.kp_seg2_new import MethodKpSegTwoNew


class YouWen(object):
    def __init__(self):
        super(YouWen, self).__init__()
        if configure.config.pointer_num == 2:
            self.method = MethodKpSegTwoNew()
        else:
            self.method = MethodKpSegOneNew()

    def __call__(self, img, filename, kp, seg):
        return self.method(img, filename, kp, seg)
