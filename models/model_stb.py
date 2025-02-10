import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import os

# from models.detectors.YOLOv5 import DetectMultiBackend
from models.backbone.convnext_mhsa import convnext_tiny
from models.decoder.SimDR_2 import PoseNet
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

        self.joint_shift_num = 35

        self.posenet_single = PoseNet(self.joint_num, self.joint_shift_num)

        self.simdr_single_loss_2d = KLDiscretLoss()
        self.simdr_single_loss_z = KLDiscretLoss()

        self.freeze(self.backbone)



    def forward(self, x, targets_singles, targets_weights_singles,
                #targets_js_single, targets_js_valid_singles,
                mode):
        # print(targets_singles.device)
        x = self.backbone(x)

        joint_single_2d, joint_single_z = self.posenet_single(x)


        if mode == 'train':
            loss = {}
            loss['simdr_single_2d'] = self.simdr_single_loss_2d(joint_single_2d, targets_singles[:,:, :2], targets_weights_singles)
            loss['simdr_single_z'] = self.simdr_single_loss_z(torch.unsqueeze(joint_single_z, 2), torch.unsqueeze(targets_singles[:,:, 2], 2), targets_weights_singles)
            # loss['single_bonedir'] = self.single_bonedir_loss(single_bonedir, targets_js_single, targets_js_valid_singles) * 0.00001

            return loss

        elif mode == 'test':
            out = {}
            joint_single = torch.cat((joint_single_2d, torch.unsqueeze(joint_single_z, 2)), dim=2)

            out['joint_simdr_single'] = joint_single

            return out


    def freeze(self, subnet):
        for p in subnet.parameters():
            p.requires_grad = False


def get_model_stb():
    model = Model()
    return model