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
        self.adaptation_type = '1x1conv'
        # self.bbox_feat_adaptation = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        #   self.cls_adaptation = nn.Linear(1024, 1024)
        #   self.reg_adaptation = nn.Linear(1024, 1024)
        self.channel_wise_adaptation = nn.ModuleList([
            nn.Linear(stu_channel[0], tea_channel[0]),
            nn.Linear(stu_channel[1], tea_channel[1]),
            nn.Linear(stu_channel[2], tea_channel[2])
        ])

        self.spatial_wise_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        ])

        #   self.roi_adaptation_layer = nn.Conv2d(256, 256, kernel_size=1)
        if self.adaptation_type == '3x3conv':
            #   3x3 conv
            self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(stu_channel[0], tea_channel[0], kernel_size=3, stride=1, padding=1),
                nn.Conv2d(stu_channel[1], tea_channel[1], kernel_size=3, stride=1, padding=1),
                nn.Conv2d(stu_channel[2], tea_channel[2], kernel_size=3, stride=1, padding=1)
            ])
        if self.adaptation_type == '1x1conv':
            #   1x1 conv
            self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(stu_channel[0], tea_channel[0], kernel_size=1, stride=1, padding=0),
                nn.Conv2d(stu_channel[1], tea_channel[1], kernel_size=1, stride=1, padding=0),
                nn.Conv2d(stu_channel[2], tea_channel[2], kernel_size=1, stride=1, padding=0)
            ])

        if self.adaptation_type == '3x3conv+bn':
            #   3x3 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(stu_channel[0], tea_channel[0], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(stu_channel[1], tea_channel[1], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(stu_channel[2], tea_channel[2], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])
        if self.adaptation_type == '1x1conv+bn':
            #   1x1 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(stu_channel[0], tea_channel[0], kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(stu_channel[1], tea_channel[1], kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(stu_channel[2], tea_channel[2], kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])

        self.student_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=stu_channel[0], inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=stu_channel[1], inter_channels=64, downsample_stride=2),
                NonLocalBlockND(in_channels=stu_channel[2])
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                NonLocalBlockND(tea_channel[0], inter_channels=64, downsample_stride=4),
                NonLocalBlockND(tea_channel[1], inter_channels=64, downsample_stride=2),
                NonLocalBlockND(tea_channel[2])
            ]
        )
        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(stu_channel[0], tea_channel[0], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(stu_channel[1], tea_channel[1], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(stu_channel[2], tea_channel[2], kernel_size=1, stride=1, padding=0)
        ])

        # self.init_stu_adap(0,0.0001,True)
    def compute_kd_loss(self,s_feats,t_feats):
        x = s_feats
        t = 0.1
        s_ratio = 1.0
        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0
        kd_nonlocal_loss = 0

        #   for channel attention
        c_t = 0.1
        c_s_ratio = 1.0

        for _i in range(len(t_feats)):
            t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [1], keepdim=True)
            size = t_attention_mask.size()
            t_attention_mask = t_attention_mask.view(x[0].size(0), -1)
            t_attention_mask = torch.softmax(t_attention_mask / t, dim=1) * size[-1] * size[-2]
            t_attention_mask = t_attention_mask.view(size)

            s_attention_mask = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
            size = s_attention_mask.size()
            s_attention_mask = s_attention_mask.view(x[0].size(0), -1)
            s_attention_mask = torch.softmax(s_attention_mask / t, dim=1) * size[-1] * size[-2]
            s_attention_mask = s_attention_mask.view(size)

            c_t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
            c_size = c_t_attention_mask.size()
            c_t_attention_mask = c_t_attention_mask.view(x[0].size(0), -1)  # 2 x 256
            c_t_attention_mask = torch.softmax(c_t_attention_mask / c_t, dim=1) * 256
            c_t_attention_mask = c_t_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

            c_s_attention_mask = self.channel_wise_adaptation[_i](torch.mean(torch.abs(x[_i]), [2, 3]))  # 2 x 256 x 1 x1
            # c_size = c_s_attention_mask.size()
            c_s_attention_mask = c_s_attention_mask.view(x[0].size(0), -1)  # 2 x 256
            c_s_attention_mask = torch.softmax(c_s_attention_mask / c_t, dim=1) * 256
            c_s_attention_mask = c_s_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

            sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
            sum_attention_mask = sum_attention_mask.detach()
            c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
            c_sum_attention_mask = c_sum_attention_mask.detach()


            kd_feat_loss += dist2(t_feats[_i], self.adaptation_layers[_i](x[_i]), attention_mask=sum_attention_mask,
                                    channel_attention_mask=c_sum_attention_mask) * 7e-5 * 6
            kd_channel_loss += torch.dist(torch.mean(t_feats[_i], [2, 3]),
                                            self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3 * 6
            t_spatial_pool = torch.mean(t_feats[_i], [1]).view(t_feats[_i].size(0), 1, t_feats[_i].size(2),
                                                                t_feats[_i].size(3))
            s_spatial_pool = torch.mean(x[_i], [1]).view(x[_i].size(0), 1, x[_i].size(2),
                                                            x[_i].size(3))
            kd_spatial_loss += torch.dist(t_spatial_pool, self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3 * 6


        for _i in range(len(t_feats)):
            s_relation = self.student_non_local[_i](x[_i])
            t_relation = self.teacher_non_local[_i](t_feats[_i])
            kd_nonlocal_loss += torch.dist(self.non_local_adaptation[_i](s_relation), t_relation, p=2)
        losses = kd_spatial_loss.view(1)+kd_feat_loss.view(1)+kd_channel_loss.view(1)+kd_nonlocal_loss.view(1)
        loss_item = torch.cat((losses,kd_spatial_loss.view(1),kd_feat_loss.view(1),kd_channel_loss.view(1),kd_nonlocal_loss.view(1)),0)
        return losses,loss_item

    def init_compute_loss_module(self,mean,stddev):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        for i in range(len(self.adaptation_layers)):
            self.adaptation_layers[i].weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            self.spatial_wise_adaptation[i].weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            self.non_local_adaptation[i].weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation



class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True, downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :
        :
        '''

        batch_size = x.size(0)  #   2 , 128 , 40 , 40

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 25
        g_x = g_x.permute(0, 2, 1)                                  

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   
        theta_x = theta_x.permute(0, 2, 1)                                 
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        N = f.size(-1)  #   150 x 150
        f_div_C = f / N #   2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  #   2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous() #   2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
    
        




def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    diff = diff * attention_mask
    diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff