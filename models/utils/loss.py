import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import sys
sys.path.append('..')
from main.config import cfg

class KLDiscretLoss(nn.Module):
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        # self.Softmax = nn.Softmax(dim=1)
        self.criterion_ = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        # print(dec_outs.shape)
        scores = self.LogSoftmax(dec_outs)
        # labels_softmax = self.Softmax(labels)
        # loss = torch.mean(self.criterion_(scores, labels_softmax), dim=1)
        loss = torch.mean(self.criterion_(scores, labels), dim=1)
        # print(loss.shape)
        return loss

    def forward(self, output, target, target_weight):
        # print('----')

        num_joints = output.size(1)
        dim = output.shape[2]
        loss = 0

        for idx in range(num_joints):
            for d in range(dim):
                coord_d_pred = output[:, idx, d]
                coord_d_gt = target[:, idx, d]
                weight = target_weight[:, idx]
                loss += (self.criterion(coord_d_pred, coord_d_gt).mul(weight).mean())
            # print(loss)
        return loss / num_joints


class JointShiftLoss(nn.Module):
    def __init__(self):
        super(JointShiftLoss, self).__init__()

    def forward(self, predicted, target, weight):
        # bone_dir = self.getbonedirect(joint_out, cfg.bone_index)
        # # print(bone_dir.shape)
        # c1 = torch.abs(torch.sum(torch.cross(bone_dir[:, :, 0], bone_dir[:, :, 1]) * bone_dir[:, :, 2], dim=2) * hand_type[:, None])
        # c2 = torch.minimum(torch.sum(torch.cross(bone_dir[:, :, 0], bone_dir[:, :, 1]) * torch.cross(bone_dir[:, :, 1], bone_dir[:, :, 2]), dim=2),
        #                    torch.zeros((joint_out.shape[0], 5)).cuda())
        # # print(c1)
        # # print(c2)
        # return torch.mean(torch.mean(c1, dim=1) - torch.mean(c2 * hand_type[:, None], dim=1))
        assert predicted.shape == target.shape
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1).mul(weight))

    # def getbonedirect(self, seq, boneindex):
    #     bs = seq.size(0)
    #     # ss = seq.size(1)
    #     # seq = seq.view(-1, seq.size(2), seq.size(3))
    #     bone = []
    #     for index in boneindex:
    #         bone.append(seq[:, index[0]] - seq[:, index[1]])
    #     bonedirect = torch.stack(bone, 1)
    #     bonedirect = bonedirect.view(bs, 5, 3, 3)
    #     return bonedirect


class Joint3DLoss(nn.Module):
    def __init__(self):
        super(Joint3DLoss, self).__init__()

    def forward(self, joint_out, joint_gt, joint_valid):
        num_joints = joint_out.size(1)
        loss = (joint_out - joint_gt)**2 * joint_valid[:,:,None]
        return loss.mean() / num_joints



class KLDiscretLoss_Joints(nn.Module):
    def __init__(self):
        super(KLDiscretLoss_Joints, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.Softmax = nn.Softmax(dim=1)
        self.criterion_ = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        # print(dec_outs.shape)
        scores = self.LogSoftmax(dec_outs)
        labels_softmax = self.Softmax(labels)
        loss = torch.mean(self.criterion_(scores, labels_softmax), dim=1)
        # print(loss.shape)
        return loss

    def forward(self, output, target):
        num_joints = output.size(1)
        dim = output.shape[2]
        loss_list = torch.zeros((num_joints))

        for idx in range(num_joints):
            for d in range(dim):
                coord_d_pred = output[:, idx, d]
                coord_d_gt = target[:, idx, d]
                loss_list[idx] += (self.criterion(coord_d_pred, coord_d_gt).mean())

        return loss_list.min()
        # return loss_list


class InterLoss_Joints(nn.Module):
    def __init__(self):
        super(InterLoss_Joints, self).__init__()

    def forward(self, joint_out, joint_gt):
        num_joints = joint_out.shape[1]
        dim = joint_out.shape[2]
        loss_list = torch.zeros((num_joints))

        for idx in range(num_joints):
            for d in range(dim):
                coord_d_pred = joint_out[:, idx, d]
                coord_d_gt = joint_gt[:, idx, d]
                loss_list[idx] += torch.sum((coord_d_pred - coord_d_gt) ** 2)

        return loss_list.min()


class BoneMapLoss(nn.Module):
    def __init__(self):
        super(BoneMapLoss, self).__init__()

    def forward(self, out, gt, valid):
        loss = (out - gt) ** 2 * valid[:, :, None, None]
        return loss.mean()