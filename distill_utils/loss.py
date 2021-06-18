# Loss functions

from traceback import print_tb
from unicodedata import decimal
import torch
from torch._C import ThroughputBenchmark, set_flush_denormal
from torch.cuda import device
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel
import numpy as np


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps



class VFLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(VFLoss, self).__init__()
        # 传递 nn.BCEWithLogitsLoss() 损失函数  must be nn.BCEWithLogitsLoss()
        self.loss_fcn = loss_fcn  #
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'mean'  # required to apply VFL to each element

    def forward(self, pred, true):

        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits

        focal_weight = true * (true > 0.0).float() + self.alpha * (pred_prob - true).abs().pow(self.gamma) * (true <= 0.0).float()
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = VFLoss(BCEcls, g), VFLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets,mask=False):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        if not mask:
            return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        sup_mask = self.get_mask(p,targets)
        return sup_mask,loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


    def get_mask(self,p, targets):
        device=targets.device
        import numpy as np
        res = []
        nl = len(p)
        for i in range(nl):
            anchor = self.anchors[i]
            pi = p[i]
            w,h = pi.shape[2],pi.shape[3]
            shift_x = np.arange(0, w)
            shift_y = np.arange(0, h)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()).to(device)
            shift_0 = torch.zeros(((shifts.shape[0]),2), device=device)
            shifts = torch.cat((shifts,shift_0),1).view(1,shifts.shape[0],-1)
            _anchors = torch.zeros((len(anchor),2), device=device)+0.5
            _anchors = torch.cat((_anchors,anchor),1).view(len(anchor),1,-1)
            all_anchors = (shifts+_anchors).view(len(anchor)*shifts.shape[1],4)
            gt_boxs = (targets[:,2:6]*w)

            batch_iou_map = []
            for j in range(pi.shape[0]):
                k = targets[:,0]==float(j)
                gt_boxs_per_image=gt_boxs[k].view(1,-1,4)
                iou = bbox_overlaps_batch(all_anchors,gt_boxs_per_image).view(len(anchor),h,w,gt_boxs_per_image.shape[1]).permute((1,2,0,3))
                mask_per_im = torch.zeros([h, w], dtype=torch.int64,device=device)
                for jj in range(gt_boxs_per_image.shape[1]):
                    per_im_iou = iou[...,jj]
                    max_iou = torch.max(per_im_iou)*0.5
                    k = (per_im_iou>=max_iou)
                    mask_per_im +=torch.sum(k,2)
                    jj=jj
                batch_iou_map.append(mask_per_im.view(1,w,h))
            batch_iou_map = torch.cat(batch_iou_map,0)
            res.append(batch_iou_map)
        return res


            # for 
            

def bbox_overlaps_batch(anchors_, gt_boxes_):
    # main reference by https://github.com/twangnh/Distilling-Object-Detectors
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    anchors = torch.zeros_like(anchors_)
    gt_boxes = torch.zeros_like(gt_boxes_)
    anchors[:,0] = anchors_[:,0]-anchors_[:,2]/2.0
    anchors[:,1] = anchors_[:,1]-anchors_[:,3]/2.0
    anchors[:,2] = anchors_[:,0]+anchors_[:,2]/2.0
    anchors[:,3] = anchors_[:,1]+anchors_[:,3]/2.0
    gt_boxes[:,:,0] = gt_boxes_[:,:,0]-gt_boxes_[:,:,2]/2.0
    gt_boxes[:,:,1] = gt_boxes_[:,:,1]-gt_boxes_[:,:,3]/2.0
    gt_boxes[:,:,2] = gt_boxes_[:,:,0]+gt_boxes_[:,:,2]/2.0
    gt_boxes[:,:,3] = gt_boxes_[:,:,1]+gt_boxes_[:,:,3]/2.0
    batch_size = gt_boxes.size(0)

    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:,:,:4].contiguous()


        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    return overlaps

def compute_sup_loss(stu_feature,tea_feature,sup_mask,weight_):
    # Proposed by the paper ：
    # Distilling Object Detectors with Fine-grained Feature Imitation
    mask_list=[]
    for i in [0,2,1]:
        mask = (sup_mask[i] > 0).float().unsqueeze(1)
        mask_list.append(mask)
    loss_item = []
    total_loss = torch.zeros(1,device=stu_feature[0].device)
    for i in range(len(stu_feature)):
        norms = mask_list[i].sum()
        loss = ((torch.pow(tea_feature[i] - stu_feature[i], 2) * mask_list[i]).sum()*weight_ / norms).view(1)
        total_loss+=loss
        loss_item.append(loss)
    loss_item = torch.cat((total_loss,torch.cat(loss_item,0)))
    return total_loss,loss_item

def compute_sup_loss_2(stu_feature,tea_feature,with_mask=False):
    # Proposed by the paper ：
    # IMPROVE OBJECT DETECTION WITH FEATURE - BASED KNOWLEDGE DISTILLATION : TOWARDS A CCURATE AND EFFICIENT DETECTORS
    import torch.nn.functional as F 
    device = stu_feature[0].device
    bs = stu_feature[0].shape[0]
    loss_func = nn.MSELoss()
    total_loss = torch.zeros(1).to(device)
    loss_item = []

    for i in range(len(stu_feature)):
        # compute the spatial mask

        stu_feature_spa = stu_feature[i].view(bs,stu_feature[i].shape[2]*stu_feature[i].shape[2],-1).contiguous()  
        tea_feature_spa = tea_feature[i].view(bs,tea_feature[i].shape[2]*tea_feature[i].shape[2],-1).contiguous()

        stu_mask_spa = F.avg_pool1d(stu_feature_spa,stu_feature_spa.shape[2]).view(bs,stu_feature[i].shape[2],stu_feature[i].shape[2])
        tea_mask_spa = F.avg_pool1d(tea_feature_spa,tea_feature_spa.shape[2]).view(bs,stu_feature[i].shape[2],stu_feature[i].shape[2])

        loss_mask_spa = loss_func(stu_mask_spa,tea_mask_spa).view(1)         # loss_mask_spa

        # compute the channel mask
        stu_mask_channel = F.avg_pool2d(stu_feature[i],stu_feature[i].shape[-1])
        tea_mask_channel= F.avg_pool2d(tea_feature[i],tea_feature[i].shape[-1])

        loss_mask_channel =loss_func(stu_mask_channel,tea_mask_channel).view(1)                # loss_mask_channel

        loss_item.append(loss_mask_channel+loss_mask_spa)
        total_loss = total_loss +loss_mask_channel+loss_mask_spa
        # print(loss)
    loss_item = torch.cat((total_loss,torch.cat(loss_item,0)))
    return total_loss,loss_item


def compute_sup_loss_3(tech_out,stu_out,weights):
    # print(tech_out.shape)
    loss =torch.zeros(1, device=tech_out[0].device)
    for i, pi in enumerate(tech_out):
        # t_location = pi[...,0:4]
        t_x = pi[...,0]
        t_y = pi[...,1]
        t_w = pi[...,2]
        t_h = pi[...,3]
        t_obj = pi[...,4]
        t_cls = pi[...,5:]
        # s_location = stu_out[i][...,0:4]
        s_x = stu_out[i][...,0]
        s_y = stu_out[i][...,1]
        s_w = stu_out[i][...,2]
        s_h = stu_out[i][...,3]
        s_obj = stu_out[i][...,4]
        s_cls = stu_out[i][...,5:]

        distill_reg_loss = obj_weighted_reg(s_x, s_y, s_w, s_h, t_x, t_y,
                                            t_w, t_h, t_obj)
        distill_cls_loss = obj_weighted_cls(s_cls, t_cls, t_obj)*0.5
        distill_obj_loss = obj_loss(s_obj, t_obj)*0.05
        distill_loss = distill_reg_loss+distill_cls_loss+distill_obj_loss
        print(distill_reg_loss,distill_cls_loss,distill_obj_loss)
        loss +=distill_loss*weights
    return loss
    
def obj_weighted_reg(sx, sy, sw, sh, tx, ty, tw, th, tobj):
    # print()
    loss_F= torch.nn.BCEWithLogitsLoss(reduction='none')
    loss_x = loss_F(sx,torch.sigmoid(tx))
    loss_y = loss_F(sy,torch.sigmoid(ty))
    loss_w = torch.abs(sw - tw)
    loss_h = torch.abs(sh - th)
    loss = loss_x+loss_y+loss_w+loss_h
    weighted_loss = torch.mean(loss * torch.sigmoid(tobj))
    return weighted_loss

def obj_weighted_cls(scls, tcls, tobj):
    loss_F = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss= loss_F(scls, torch.sigmoid(tcls))
    loss = torch.mean(loss,dim=4)
    weighted_loss = torch.mean(loss*torch.sigmoid(tobj))
    return weighted_loss

def obj_loss(sobj, tobj):
    obj_mask = (tobj>0).float()
    loss_f =torch.nn.BCEWithLogitsLoss(reduction='none')
    loss = torch.mean(loss_f(sobj, obj_mask))
    return loss
