import configure
from meter import *


class Bjds2(object):
    def __init__(self, ini_path):
        DANGWEI_LIST = ['dangwei_1', 'dangwei_4']
        JCQ_LIST = ['JCQ_4_POINT', 'JCQ_5_POINT', 'JCQ_6_POINT', 'JCQ_7_POINT', 'JCQ_8_POINT', "JCQ_ALL"]
        SF6_LIST = ['1_2_SF6', '9_SF6', '10_SF6', '12_SF6', 'SF6_2', 'SF6_3']
        YOUWEI_LIST = ['youwei_1', 'youwei_2', 'youwei_3', 'youwei_7', 'youwei_9', 'youwei_10', 'youwei_11',
                       'youwei_12', 'youwei_13',
                       'youwei_14', 'youwei_15', 'youwei_16', 'youwei_17', 'youwei_18', 'youwei_19', 'youwei_20',
                       'youwei_21', 'youwei_22',
                       'youwei_23', 'youwei_24', 'youwei_25', 'youwei_27', 'youwei_28', 'youwei_29', 'youwei_30',
                       'youwei_31', 'youwei_32',
                       'youwei_33', 'youwei_34', 'youwei_35', '32yweij_YZF3', '52_bj', 'wasi_1', 'dianliu_dianya']
        YOUWEN_LIST = ['23ywenj_BWY', '26_raozu', '27_B4A-804AJ', '48_bj', '55ywenj_LEAD', 'youwen_1',
                       'youwen_2', 'youwen_7']
        YOUYA_LIST = ['yali_2', 'yali_3', 'youya_1', 'youya_2', 'youya_3']
        YOULIU_LIST = ['youliu_1', 'youliu_2', 'youliu_3']

        configure.config.set_arg(ini_path, True)
        self.ini_path = ini_path
        print(configure.config.name)
        self.name = configure.config.name
        self.scale_carve = configure.config.scale_carve

        if configure.config.name in DANGWEI_LIST:
            self.recognition = DangWei()
        elif configure.config.name in JCQ_LIST:
            self.recognition = JCQ()
        elif configure.config.name in SF6_LIST:
            self.recognition = SF6()
        elif configure.config.name in YOUWEI_LIST:
            self.recognition = YouWei()
        elif configure.config.name in YOUWEN_LIST:
            self.recognition = YouWen()
        elif configure.config.name in YOUYA_LIST:
            self.recognition = YouYa()
        elif configure.config.name in YOULIU_LIST:
            self.recognition = YouLiu()
        else:
            assert False, print('This meter is not supported!')

    def process(self, cv2_mat, filename, kp=None, seg=None):
        configure.config.set_arg(self.ini_path, True)
        return self.recognition(cv2_mat, filename, kp, seg)
