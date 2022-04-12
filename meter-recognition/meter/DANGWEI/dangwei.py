import configure
from method.kp_seg1_old import MethodKpSegOneOld
from method.kp_seg1_new import MethodKpSegOneNew


class DangWei(object):
    def __init__(self):
        super(DangWei, self).__init__()
        if configure.config.name in ['dangwei_4']:
            self.method = MethodKpSegOneNew()
        if configure.config.name in ['dangwei_1']:
            self.method = MethodKpSegOneOld()

    def __call__(self, img, filename, kp, seg):
        return self.method(img, filename, kp, seg)
