from method.kp_seg1_old import MethodKpSegOneOld


class YouYa(object):
    def __init__(self):
        super(YouYa, self).__init__()
        self.method = MethodKpSegOneOld()

    def __call__(self, img, filename, kp, seg):
        return self.method(img, filename, kp, seg)
