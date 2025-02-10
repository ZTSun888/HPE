import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import os

# from models.detectors.YOLOv5 import DetectMultiBackend
from models.backbone.convnext import convnext_tiny
# from models.backbone.convnext_mhsa import convnext_tiny
from models.decoder.SimDR_bone import PoseNet, PoseNet_z
from main.config import cfg
from models.utils.loss import KLDiscretLoss, JointShiftLoss, BoneMapLoss

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # self.detector = DetectMultiBackend(yolo_weights, device)
        # self.detector.eval()
        self.joint_num = 21
        self.backbone = convnext_tiny(model_path='../models/backbone/convnext_tiny_1k_224_ema.pth', convnext_pretrained=True)
        # self.backbone = convnext_tiny(convnext_pretrained=False)

        self.joint_shift_num = 20

        self.posenet_single = PoseNet(self.joint_num, self.joint_shift_num)
        self.posenet_inter = PoseNet(self.joint_num * 2, self.joint_shift_num * 2)
        self.posenet_z_single = PoseNet_z(self.joint_num, self.joint_shift_num)
        self.posenet_z_inter = PoseNet_z(self.joint_num * 2, self.joint_shift_num * 2)

        self.freeze(self.posenet_z_single)
        self.freeze(self.posenet_z_inter)

        self.simdr_single_loss_2d = KLDiscretLoss()
        self.simdr_inter_loss_2d = KLDiscretLoss()
        self.single_bonemap_loss = BoneMapLoss()
        self.inter_bonemap_loss = BoneMapLoss()

        self.simdr_single_loss_3d = KLDiscretLoss()
        self.simdr_inter_loss_3d = KLDiscretLoss()


    def forward(self, x, targets_singles, targets_inters, targets_weights_singles, targets_weights_inters,
                targets_bone_map_singles, targets_bone_map_inters,
                targets_bone_weight_singles, targets_bone_weight_inters,
                mode):
        x= self.backbone(x)

        joint_single_2d, single_bone_map, feature_vector_single, pred_xy_single = self.posenet_single(x)
        joint_inter_2d, inter_bone_map, feature_vector_inter, pred_xy_inter = self.posenet_inter(x)
        # joint_single_3d = self.posenet_z_single(pred_xy_single, single_bone_map, feature_vector_single)
        # joint_inter_3d = self.posenet_z_inter(pred_xy_inter, inter_bone_map, feature_vector_inter)


        if mode == 'train':
            loss = {}
            loss['simdr_single_2d'] = self.simdr_single_loss_2d(joint_single_2d, targets_singles[:,:, :2], targets_weights_singles) * 100
            loss['simdr_inter_2d'] = self.simdr_inter_loss_2d(joint_inter_2d, targets_inters[:,:, :2], targets_weights_inters) * 100
            loss['bone_map_single'] = self.single_bonemap_loss(single_bone_map, targets_bone_map_singles, targets_bone_weight_singles)
            loss['bone_map_inter'] = self.inter_bonemap_loss(inter_bone_map, targets_bone_map_inters, targets_bone_weight_inters)

            # loss['simdr_single_3d'] = self.simdr_single_loss_3d(joint_single_3d, targets_singles, targets_weights_singles) * 100
            # loss['simdr_inter_3d'] = self.simdr_inter_loss_3d(joint_inter_3d, targets_inters,targets_weights_inters) * 100


            return loss

        elif mode == 'test':
            out = {}
            joint_single = torch.cat((joint_single_2d, torch.unsqueeze(joint_single_z, 2)), dim=2)
            joint_inter = torch.cat((joint_inter_2d, torch.unsqueeze(joint_inter_z, 2)), dim=2)
            out['joint_simdr_single'] = joint_single
            out['joint_simdr_inter'] = joint_inter
            return out


    def freeze(self, subnet):
        for p in subnet.parameters():
            p.requires_grad = False


def get_model():
    model = Model()
    return model