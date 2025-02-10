import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import os

# from models.detectors.YOLOv5 import DetectMultiBackend
# from models.backbone.convnext import convnext_tiny
from models.backbone.convnext_mhsa import convnext_tiny
# from models.backbone.convnext_scconv import convnext_tiny
from models.decoder.SimDR_2 import PoseNet
from models.backbone.swin_transformer import swin_tiny
from models.backbone.resnet import ResNetBackbone
from models.backbone.shufflenet import ShuffleNet
from main.config import cfg
from models.utils.loss import KLDiscretLoss, JointShiftLoss

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # self.detector = DetectMultiBackend(yolo_weights, device)
        # self.detector.eval()
        self.joint_num = 21
        self.backbone = convnext_tiny(model_path='../models/backbone/convnext_tiny_1k_224_ema.pth', convnext_pretrained=True)
        # self.backbone = convnext_tiny(convnext_pretrained=False)
        # self.backbone = ResNetBackbone(50)
        # self.backbone = ShuffleNet()
        # self.backbone = swin_tiny(model_path='../models/backbone/swin_tiny_patch4_window7_224.pth', pretrained=True)

        self.joint_shift_num = 35

        self.posenet_single = PoseNet(self.joint_num, self.joint_shift_num)
        self.posenet_inter = PoseNet(self.joint_num * 2, self.joint_shift_num * 2)

        # self.freeze(self.backbone)
        # self.freeze(self.posenet_single)
        # self.freeze(self.posenet_inter)

        self.simdr_single_loss_2d = KLDiscretLoss()
        self.simdr_inter_loss_2d = KLDiscretLoss()
        self.simdr_single_loss_z = KLDiscretLoss()
        self.simdr_inter_loss_z = KLDiscretLoss()

        # self.simdr_opt_single_loss = KLDiscretLoss()
        self.single_bonedir_loss = JointShiftLoss()
        # self.simdr_opt_inter_loss = KLDiscretLoss()
        self.inter_bonedir_loss = JointShiftLoss()


    def forward(self, x, targets_singles, targets_inters, targets_weights_singles, targets_weights_inters,
                mode):
        # print(targets_singles.device)
        x = self.backbone(x)
        latent_vector = x
        # joint_single_2d, joint_single_z, single_bonedir, single_attn = self.posenet_single(x)
        # joint_inter_2d, joint_inter_z, inter_bonedir, inter_attn= self.posenet_inter(x)
        joint_single_2d, joint_single_z = self.posenet_single(x)
        joint_inter_2d, joint_inter_z = self.posenet_inter(x)

        if mode == 'train':
            loss = {}
            loss['simdr_single_2d'] = self.simdr_single_loss_2d(joint_single_2d, targets_singles[:,:, :2], targets_weights_singles)*100
            loss['simdr_inter_2d'] = self.simdr_inter_loss_2d(joint_inter_2d, targets_inters[:,:, :2], targets_weights_inters)*100
            loss['simdr_single_z'] = self.simdr_single_loss_z(torch.unsqueeze(joint_single_z, 2), torch.unsqueeze(targets_singles[:,:, 2], 2), targets_weights_singles)*100
            loss['simdr_inter_z'] = self.simdr_inter_loss_z(torch.unsqueeze(joint_inter_z, 2), torch.unsqueeze(targets_inters[:,:, 2], 2), targets_weights_inters)*100

            # loss['single_bonedir'] = self.single_bonedir_loss(single_bonedir, targets_js_single, targets_js_valid_singles) * 0.0001
            # loss['simdr_opt_single'] = self.simdr_opt_single_loss(opt_single, targets_singles, targets_weights_singles)
            # loss['inter_bonedir'] = self.single_bonedir_loss(inter_bonedir, targets_js_inter,targets_js_valid_inters) * 0.0001
            # loss['simdr_opt_inter'] = self.simdr_opt_single_loss(opt_inter, targets_inters, targets_weights_inters)


            # loss['js_single'] = self.js_single_loss(bonejs_single, targets_js_single, targets_js_valid_singles)
            # loss['js_inter'] = self.js_inter_loss(bonejs_inter, targets_js_inter, targets_js_valid_inters)

            return loss

        elif mode == 'test':
            out = {}
            joint_single = torch.cat((joint_single_2d, torch.unsqueeze(joint_single_z, 2)), dim=2)
            joint_inter = torch.cat((joint_inter_2d, torch.unsqueeze(joint_inter_z, 2)), dim=2)
            out['joint_simdr_single'] = joint_single
            out['joint_simdr_inter'] = joint_inter
            # out['single_attn'] = single_attn
            # out['inter_attn'] = inter_attn
            # return out, latent_vector
            return out


    def freeze(self, subnet):
        for p in subnet.parameters():
            p.requires_grad = False


def get_model():
    model = Model()
    return model