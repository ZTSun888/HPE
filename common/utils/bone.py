import torch
import numpy as np
import sys
sys.path.append('..')
from main.config import cfg

def getbonejs(seq, joint_valid):
    boneindexs = cfg.shift_index
    js_bone = np.zeros((cfg.joint_shift_num, 3))
    js_valid = np.zeros((cfg.joint_shift_num))
    for i in range (0, len(boneindexs)):
        boneindex = boneindexs[i]
        js_bone[i]= seq[boneindex[0]] - seq[boneindex[1]]
        if (joint_valid[[boneindex[0]]] and joint_valid[[boneindex[1]]]):
            js_valid[i] = 1
    # bone = torch.stack(bone,1)
    # bone = bone.view(bs,ss, bone.size(1),3)
    return js_bone, js_valid


def getbonevec(seq, joint_valid):
    boneindexs = cfg.bone_index
    js_bone = np.zeros((cfg.bone_num, 3))
    js_valid = np.zeros((cfg.bone_num))
    for i in range (0, len(boneindexs)):
        boneindex = boneindexs[i]
        js_bone[i]= seq[boneindex[0]] - seq[boneindex[1]]
        if (joint_valid[[boneindex[0]]] and joint_valid[[boneindex[1]]]):
            js_valid[i] = 1
    return js_bone, js_valid
