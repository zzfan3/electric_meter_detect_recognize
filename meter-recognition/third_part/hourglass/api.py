import configure
import os
import torch
from ..common import letter_box_image
from .model import MeterNet
from .utils import get_preds_fromhm


class KeyPoint(object):
    def __init__(self, feature_point_num=False, adjust_model_path=False):
        super(KeyPoint, self).__init__()
        if not feature_point_num:
            feature_point_num = configure.config.feature_point_num
        if not adjust_model_path:
            adjust_model_path = configure.config.adjust_model_path
        self.feature_point_num = feature_point_num
        self.checkpoint_model_path = adjust_model_path
        self.num_hg = 3
        self.gpu_num = 1
        # self.gpu_num = 0
        self.model = MeterNet(self.num_hg, self.feature_point_num)
        # self.checkpoint_dir = 'adjust/checkpoints_55ywenj_LEAD_7point'

        if self.gpu_num == 0:
            self.cuda = False
        else:
            self.cuda = True

        if os.path.exists(self.checkpoint_model_path):
            self.model.load_state_dict(
                torch.load(self.checkpoint_model_path, map_location=lambda storage, loc: storage)['state_dict'])
            print("Load hourglass pretrained model success!")
        else:
            assert False, "Load hourglass pretrained model fail!"

    def __call__(self, input_image, resolution, value, mean_std, info=True):
        # new_img, ratio, offset_w, offset_h = letter_box_image(input_image, 640, 640)
        new_img = letter_box_image(input_image, resolution, resolution, value=value)
        inp = torch.from_numpy(new_img).float().div(255.0)  # .unsqueeze_(0)
        # ---------------------------减均值除以方差---------------------------
        if mean_std == True:
            inp = inp - torch.mean(inp)
            inp = inp / torch.std(inp)
        inp = inp.permute((2, 0, 1))
        inp = inp.unsqueeze(0)

        if self.cuda:
            inp = inp.cuda()
            self.model.cuda()
        self.model.eval()

        with torch.no_grad():
            out = self.model(inp)[-1].data.cpu()

        pts, pts_img = get_preds_fromhm(out, None, None)
        pts, pts_img = pts.view(self.feature_point_num, 2) * 4, pts_img.view(
            self.feature_point_num, 2)  # hwzhao  pts, pts_img = pts.view(11, 2) * 4, pts_img.view(11, 2) # xiugai 1
        pts = pts.numpy()
        return pts, new_img
