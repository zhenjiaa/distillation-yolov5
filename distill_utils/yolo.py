from torch._C import set_flush_denormal
from torch.cuda import init
from models.yolo import Model
import torch.nn as nn
import torch
# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None



class dist_model(Model):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None,stu_channel=[128,256,128],tea_channel = [192,384,192]):
        Model.__init__(self, cfg, ch=3, nc=nc, anchors=anchors)
        self.stu_channel = stu_channel
        self.tea_channel = tea_channel
        self._stu_feature_adap = nn.ModuleList(nn.Sequential(nn.Conv2d(self.stu_channel[i],self.tea_channel[i],kernel_size=3,padding=1),nn.ReLU()) \
             for i in range(len(self.tea_channel)))
        # self.init_stu_adap(0,0.0001,True)
    def init_stu_adap(self,mean,stddev,truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        for i in range(len(self._stu_feature_adap)):
            if truncated:
                self._stu_feature_adap[i][0].weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                self._stu_feature_adap[i][0].weight.data.normal_(mean, stddev)
                self._stu_feature_adap[i][0].bias.data.zero_()
        


    # def forward(self, x, augment=False, profile=False,feature_layer=[4,14,10]):
    #     if augment:
    #         img_size = x.shape[-2:]  # height, width
    #         s = [1, 0.83, 0.67]  # scales
    #         f = [None, 3, None]  # flips (2-ud, 3-lr)
    #         y = []  # outputs
    #         for si, fi in zip(s, f):
    #             xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
    #             yi = self.forward_once(xi,feature_layer=[4,14,10])[0]  # forward
    #             # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
    #             yi[..., :4] /= si  # de-scale
    #             if fi == 2:
    #                 yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
    #             elif fi == 3:
    #                 yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
    #             y.append(yi)
    #         return torch.cat(y, 1), None  # augmented inference, train
    #     else:
    #         return self.forward_once(x, profile,feature_layer=[4,14,10])  # single-scale inference, train

    # def forward_once(self, x, profile=False,feature_layer=[4,14,10]):
    #     y, dt = [], []  # outputs
    #     rs = []
    #     for i,m in enumerate(self.model):
    #         if m.f != -1:  # if not from previous layer
    #             x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

    #         if profile:
    #             o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
    #             t = time_synchronized()
    #             for _ in range(10):
    #                 _ = m(x)
    #             dt.append((time_synchronized() - t) * 100)
    #             if m == self.model[0]:
    #                 logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
    #             logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

    #         x = m(x)  # run
    #         if  i in feature_layer:
    #             # print(x.shape)
    #             rs.append(x)
    #         # print(m.i)
    #         y.append(x if m.i in self.save else None)  # save output

