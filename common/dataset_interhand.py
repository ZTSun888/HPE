import os

import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp

from pycocotools.coco import COCO
import sys
sys.path.append('..')
# from common.utils.coco_diy import COCO
from common.utils.load_data import load_data
from common.utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from common.utils.transforms import world2cam, cam2pixel, pixel2cam
import json
from torchvision import transforms
from common.utils.vis import vis_keypoints, vis_kp_bbox, vis_simdr,vis_attn_matrix
from common.utils.bone import getbonejs, getbonevec
import math

import matplotlib.pyplot as plt
import random

#
# class cfg:
#     output_hm_shape = (64, 64, 64)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, annot_subset):
        self.mode = mode  # train, test, val
        self.annot_subset = annot_subset  # all, human_annot, machine_annot
        self.img_path = '../data/InterHand2.6M_new/images'
        self.annot_path = '../data/InterHand2.6M_new/human_annot'

        # self.transform = transforms.ToTensor()
        self.joint_num = 21  # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num * 2)}
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num * 2)
        # self.joint_shift_num = 35
        # self.js_type = {'right':np.arange(0, self.joint_shift_num), 'left': np.arange(self.joint_shift_num, self.joint_shift_num*2)}
        self.bone_num = 20
        self.bone_type = {'right': np.arange(0, self.bone_num),
                        'left': np.arange(self.bone_num, self.bone_num * 2)}
        self.bone_index = cfg.bone_index


        self.datalist = []

        self.datalist_rh = []
        self.datalist_lh = []
        self.datalist_ih = []
        self.datalist_th = []

        self.sequence_names = []
        self.output_hm_shape = cfg.output_hm_shape
        self.input_img_shape = cfg.input_img_shape
        # self.simdr_split_ratio = cfg.simdr_split_ratio
        self.sigma = cfg.sigma
        self.bbox_3d_size = cfg.bbox_3d_size

        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((self.input_img_shape[1], self.input_img_shape[0])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # load annotation
        print("Load annotation from  " + osp.join(self.annot_path, self.annot_subset))
        db = COCO(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data.json'))
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3'
                                                                                      'd.json')) as f:
            joints = json.load(f)

        print("Get bbox from groundtruth annotation")

        i = 0
        # rh = 0
        # lh = 0
        # th = 0
        # ih = 0
        for aid in db.anns.keys():

            i += 1
            if(i == 1000):
                break
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]

            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']

            img_path = osp.join(self.img_path, self.mode, img['file_name'])

            bonemap_json_dir = osp.join(self.annot_path, self.mode, img['file_name'])
            bonemap_json_path = bonemap_json_dir.split('.jpg')[0]+'.json'

            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
            joint_cam = world2cam(joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

            joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(self.joint_num * 2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]

            # bone_valid = self.calBonesValid(joint_valid)

            hand_type = ann['hand_type']
            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

            # img_width, img_height = img['width'], img['height']
            bboxs = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
            img_width, img_height = img['width'], img['height']
            abs_depth = {'right': joint_cam[self.root_joint_idx['right'], 2],
                             'left': joint_cam[self.root_joint_idx['left'], 2]}

            # print(bbox)
            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bboxs': bboxs, 'joint': joint,
                    'hand_type': hand_type, 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth,
                    'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
                    'is_two': False,
                    'bonemap_json_path': bonemap_json_path}

            if hand_type == 'right':
                # rh += 1
                # if (rh > 50):
                #     continue
                bboxs[0] = process_bbox(bboxs[0], (img_height, img_width))
                data['bbox'] = bboxs[0]
                self.datalist_rh.append(data)
            elif hand_type == 'left':
                # lh += 1
                # if (lh > 50):
                #     continue
                bboxs[0] = process_bbox(bboxs[0], (img_height, img_width))
                data['bbox'] = bboxs[0]
                self.datalist_lh.append(data)
            elif hand_type == 'two':
                # continue
                # th += 1
                # if (th > 50):
                #     continue
                # self.datalist_th.append(data)

                joint_valid_right = np.zeros((self.joint_num * 2), dtype=np.float32)
                joint_valid_right[self.joint_type['right']] = joint_valid[self.joint_type['right']]
                joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid_right}
                bboxs[0] = process_bbox(bboxs[0], (img_height, img_width))
                data_right = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bboxs[0],
                              'joint': joint,
                              'hand_type': 'right', 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth,
                              'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
                              'bonemap_json_path': bonemap_json_path.split('.json')[0]+'_0.json'}
                # print(data_right['joint']['valid'])


                joint_valid_left = np.zeros((self.joint_num * 2), dtype=np.float32)
                joint_valid_left[self.joint_type['left']] = joint_valid[self.joint_type['left']]
                joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid_left}
                bboxs[1] = process_bbox(bboxs[1], (img_height, img_width))
                data_left = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bboxs[1],
                             'joint': joint,
                             'hand_type': 'left', 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth,
                             'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
                             'bonemap_json_path': bonemap_json_path.split('.json')[0]+'_1.json'}
                if self.mode == 'test':
                    data_right['is_two'] = True
                    data_left['is_two'] = True
                self.datalist_th.append(data_right)
                self.datalist_th.append(data_left)
            else:
                # ih += 1
                # # if not ih % 100 == 0:
                # #     continue
                # if (ih > 50):
                #     continue
                bboxs[0] = process_bbox(bboxs[0], (img_height, img_width))
                data['bbox'] = bboxs[0]
                self.datalist_ih.append(data)
            # if seq_name not in self.sequence_names:
            #     self.sequence_names.append(seq_name)


        self.datalist = self.datalist_rh + self.datalist_lh + self.datalist_th + self.datalist_ih
        # self.datalist = self.datalist_rh + self.datalist_lh + self.datalist_th
        # self.datalist = self.datalist_ih
        print('Number of annotations in right hand sequences: ' + str(len(self.datalist_rh)))
        print('Number of annotations in left hand sequences: ' + str(len(self.datalist_lh)))
        print('Number of annotations in two hand sequences: ' + str(len(self.datalist_th)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))


    # def __init__(self, cfg, mode, annot_subset):
    #     # for creating vedio-model input json
    #     self.mode = 'train'  # train, test, val
    #
    #     self.annot_subset = annot_subset  # all, human_annot, machine_annot
    #     self.img_path = '../data/InterHand2.6M_vedio/images'
    #     self.annot_path = '../data/InterHand2.6M_vedio/annotations'
    #     self.joint_num = 21  # single hand
    #     self.root_joint_idx = {'right': 20, 'left': 41}
    #     self.joint_type = {'right': np.arange(0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num * 2)}
    #     self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num * 2)
    #     self.joint_shift_num = 35
    #     self.js_type = {'right':np.arange(0, self.joint_shift_num), 'left': np.arange(self.joint_shift_num, self.joint_shift_num*2)}
    #
    #     self.output_hm_shape = cfg.output_hm_shape
    #     self.input_img_shape = cfg.input_img_shape
    #     self.sigma = cfg.sigma
    #     self.bbox_3d_size = cfg.bbox_3d_size
    #
    #     self.transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #
    #     # load annotation
    #     print("Load annotation from  " + osp.join(self.annot_path, self.annot_subset))
    #     with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data_bbox.json')) as f:
    #         dbs = json.load(f)
    #     with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
    #         cameras = json.load(f)
    #     with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
    #         joints = json.load(f)
    #
    #     print("Get bbox from groundtruth annotation")
    #
    #     i = 0
    #     ti = 0
    #     nti = 0
    #     self.idx_list = []
    #     vedio_idx = 0
    #     self.datalist = []
    #     for vedio_dict in dbs:
    #
    #         img_idx = 0
    #         vedio_path = vedio_dict['vedio_path']
    #         datalist = []
    #         db = COCO(vedio_dict['data'])
    #
    #         # x = random.randint(0, 100)
    #         # if x >= 10:
    #         #     continue
    #
    #         for aid in db.anns.keys():
    #
    #             if i >= 1:
    #                 continue
    #             i += 1
    #
    #             ann = db.anns[aid]
    #             image_id = ann['image_id']
    #             img = db.loadImgs(image_id)[0]
    #
    #             capture_id = img['capture']
    #             # if capture_id != 0:
    #             #     break
    #             seq_name = img['seq_name']
    #             cam = img['camera']
    #             frame_idx = img['frame_idx']
    #
    #             img_path = osp.join(self.img_path, self.mode, img['file_name'])
    #
    #             campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
    #                 cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
    #             focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
    #                 cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
    #             joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
    #             joint_cam = world2cam(joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
    #             joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]
    #
    #             joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(self.joint_num * 2)
    #             # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
    #             joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
    #             joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
    #
    #             hand_type = ann['hand_type']
    #             hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)
    #
    #             img_width, img_height = img['width'], img['height']
    #             bboxs = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
    #             # bbox = process_bbox(bbox, (img_height, img_width))    houxuchuli
    #             abs_depth = {'right': joint_cam[self.root_joint_idx['right'], 2], 'left': joint_cam[self.root_joint_idx['left'], 2]}
    #
    #             # print(bbox)
    #             if hand_type != 'two':
    #                 # if hand_type != 'interacting':
    #                 #     continue
    #                 # nti += 1
    #                 # if (nti+50) % 99 != 0:
    #                 #     continue
    #                 # if nti > 20000:
    #                 #     continue
    #                 bbox = bboxs[0]
    #                 bbox = process_bbox(bbox, (img_height, img_width))
    #                 cam_param = {'focal': focal, 'princpt': princpt}
    #                 joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
    #                 data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bbox, 'joint': joint,
    #                         'hand_type': hand_type, 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth,
    #                         'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
    #                         'is_two': False}
    #                 datalist.append(data)
    #                 self.idx_list.append([vedio_idx, img_idx])
    #                 img_idx += 1
    #
    #             elif hand_type == 'two':
    #                 # continue
    #                 bboxs[0] = process_bbox(bboxs[0], (img_height, img_width))
    #                 bboxs[1] = process_bbox(bboxs[1], (img_height, img_width))
    #                 cam_param = {'focal': focal, 'princpt': princpt}
    #
    #                 joint_valid_right = np.zeros((self.joint_num * 2), dtype=np.float32)
    #                 joint_valid_right[self.joint_type['right']] = joint_valid[self.joint_type['right']]
    #                 joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid_right}
    #                 data_right = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bboxs[0],
    #                                 'joint': joint,'hand_type': 'right', 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth,
    #                                 'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
    #                                 'is_two': True}
    #                 datalist.append(data_right)
    #                 self.idx_list.append([vedio_idx, img_idx])
    #                 img_idx += 1
    #
    #
    #                 joint_valid_left = np.zeros((self.joint_num * 2), dtype=np.float32)
    #                 joint_valid_left[self.joint_type['left']] = joint_valid[self.joint_type['left']]
    #                 joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid_left}
    #                 data_left = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bboxs[1],
    #                                 'joint': joint,'hand_type': 'left', 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth,
    #                                 'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
    #                                 'is_two': True}
    #                 datalist.append(data_left)
    #                 self.idx_list.append([vedio_idx, img_idx])
    #                 img_idx += 1
    #
    #
    #         vedio_list = {'vedio_path': vedio_path, 'datalist': datalist}
    #         self.datalist.append(vedio_list)
    #         vedio_idx += 1
    #
    #     print('total img num: '+ str(len(self.idx_list)))
    #     print('total video num: '+ str(len(self.datalist)))

    # def __init__(self, cfg, mode, annot_subset):
    #     # for continue train
    #     self.mode = mode  # train, test, val
    #
    #     self.annot_subset = annot_subset  # all, human_annot, machine_annot
    #
    #     self.joint_num = 21  # single hand
    #     self.root_joint_idx = {'right': 20, 'left': 41}
    #     self.joint_type = {'right': np.arange(0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num * 2)}
    #     self.skeleton = load_skeleton(osp.join('../data/InterHand2.6M/human_annot', 'skeleton.txt'), self.joint_num * 2)
    #     self.joint_shift_num = 35
    #     self.js_type = {'right':np.arange(0, self.joint_shift_num), 'left': np.arange(self.joint_shift_num, self.joint_shift_num*2)}
    #
    #     self.output_hm_shape = cfg.output_hm_shape
    #     self.input_img_shape = cfg.input_img_shape
    #     self.sigma = cfg.sigma
    #     self.bbox_3d_size = cfg.bbox_3d_size
    #
    #
    #     self.transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #
    #     self.datalist = []
    #     # self.annot_path = '../data/InterHand2.6M_new/human_annot'
    #     if self.mode == 'train':
    #         self.datalist.extend(load_data(osp.join('../data/InterHand2.6M_new/human_annot', self.mode, 'InterHand2.6M_' + self.mode + '_data.json'),
    #                                        osp.join('../data/InterHand2.6M_new/human_annot', self.mode, 'InterHand2.6M_' + self.mode + '_camera.json'),
    #                                        osp.join('../data/InterHand2.6M_new/human_annot', self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json'),
    #                                        '../data/InterHand2.6M_new/images/train',
    #                                        self.skeleton))
    #         self.datalist.extend(load_data(osp.join('../data/InterHand2.6M_new_add/annotations', self.mode, 'InterHand2.6M_' + self.mode + '_data.json'),
    #                                        osp.join('../data/InterHand2.6M_new_add/annotations', self.mode, 'InterHand2.6M_' + self.mode + '_camera.json'),
    #                                        osp.join('../data/InterHand2.6M_new_add/annotations', self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json'),
    #                                       osp.join('../data/InterHand2.6M_new_add/images', self.mode),
    #                                        self.skeleton))
    #         self.datalist.extend(load_data(osp.join('../data/InterHand2.6M_new_add/annotations', 'test','InterHand2.6M_' + 'test' + '_data.json'),
    #                                        osp.join('../data/InterHand2.6M_new_add/annotations', 'test','InterHand2.6M_' + 'test' + '_camera.json'),
    #                                        osp.join('../data/InterHand2.6M_new_add/annotations', 'test','InterHand2.6M_' + 'test' + '_joint_3d.json'),
    #                                        osp.join('../data/InterHand2.6M_new_add/images', 'test'),
    #                                        self.skeleton))
    #     else:
    #         self.datalist.extend(load_data(osp.join('../data/InterHand2.6M_vedio/annotations', self.mode,'InterHand2.6M_' + self.mode + '_data_img.json'),
    #                                        osp.join('../data/InterHand2.6M_vedio/annotations', self.mode,'InterHand2.6M_' + self.mode + '_camera.json'),
    #                                        osp.join('../data/InterHand2.6M_vedio/annotations', self.mode,'InterHand2.6M_' + self.mode + '_joint_3d.json'),
    #                                        osp.join('../data/InterHand2.6M_vedio/images', self.mode),
    #                                        self.skeleton))

    def __len__(self):
        return len(self.datalist)
    # def __len__(self):
    #     for creating vedio-model input json
        # return len(self.idx_list)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy();
        joint_img = joint['img_coord'].copy();
        joint_valid = joint['valid'].copy();
        # hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)

        # image load
        img = load_img(img_path)
        # augmentation

        joint_simdr_singles = np.zeros((self.joint_num, 3, self.output_hm_shape[0]), dtype=np.float32)
        joint_simdr_inters = np.zeros((self.joint_num * 2, 3, self.output_hm_shape[0]), dtype=np.float32)
        joint_valid_singles = np.zeros((self.joint_num), dtype=np.float32)
        joint_valid_inters = np.zeros((self.joint_num * 2), dtype=np.float32)
        # js_single = np.zeros((self.joint_shift_num, 3), dtype=np.float32)
        # js_inter = np.zeros((self.joint_shift_num * 2, 3), dtype=np.float32)
        # js_valid_single = np.zeros((self.joint_shift_num), dtype=np.float32)
        # js_valid_inter = np.zeros((self.joint_shift_num * 2), dtype=np.float32)

        img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord,
                                                                                   joint_valid,
                                                                                   hand_type, self.mode,
                                                                                   self.joint_type)
        rel_root_depth = np.array(
                    [joint_coord[self.root_joint_idx['left'], 2] - joint_coord[self.root_joint_idx['right'], 2]],
                    dtype=np.float32).reshape(1)
         # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth = transform_input_to_output_space(joint_coord, joint_valid,
                                                                                                    rel_root_depth,
                                                                                                    # root_valid,
                                                                                                    self.root_joint_idx,
                                                                                                    self.joint_type)
        joint_simdr, joint_valid = self.generate_sa_simdr(joint_coord, joint_valid)

        # joint_coord = self.simdr2coord(joint_simdr)

        img = self.transform(img.astype(np.float32))

        if hand_type == 'right' or hand_type == 'left':
            joint_simdr_singles = joint_simdr[self.joint_type[hand_type]]
            joint_valid_singles = joint_valid[self.joint_type[hand_type]]
            # js_single, js_valid_single = getbonejs(joint_coord[self.joint_type[hand_type]], joint_valid[self.joint_type[hand_type]])

            joint_valid_inters[self.joint_type[hand_type]] = joint_valid_singles
            joint_simdr_inters[self.joint_type[hand_type]] = joint_simdr_singles
            # js_inter[self.js_type[hand_type]], js_valid_inter[self.js_type[hand_type]] = getbonejs(joint_coord[self.joint_type[hand_type]], joint_valid[self.joint_type[hand_type]])
        elif hand_type == 'interacting':
            joint_simdr_inters = joint_simdr
            joint_valid_inters = joint_valid
            # js_inter[self.js_type['right']], js_valid_inter[self.js_type['right']] \
            #         = getbonejs(joint_coord[self.joint_type['right']], joint_valid[self.joint_type['right']])
            # js_inter[ self.js_type['left']], js_valid_inter[self.js_type['left']]\
            #         = getbonejs(joint_coord[self.joint_type['left']], joint_valid[self.joint_type['left']])

        # print(bbox)
        inputs = {'img': img}
        targets = {'joint_simdr_singles': joint_simdr_singles, 'joint_simdr_inters': joint_simdr_inters,
                   'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
                   # 'js_single': bone_single, 'js_inter': bone_inter}
        meta_info = {'joint_valid_singles': joint_valid_singles, 'joint_valid_inters': joint_valid_inters,
                    'inv_trans': inv_trans, 'joint_valid': joint_valid}
                     # 'js_valid_singles': bone_valid_single, 'js_valid_inters':bone_valid_inter}


        return inputs, targets, meta_info


    # def __getitem__(self, idx):
    #     # for creating vedio-model input json
    #     vedio_idx, img_idx = self.idx_list[idx]
    #     data = self.datalist[vedio_idx]['datalist'][img_idx]
    #     img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data[
    #         'hand_type'], data['hand_type_valid']
    #     joint_cam = joint['cam_coord'].copy();
    #     joint_img = joint['img_coord'].copy();
    #     joint_valid = joint['valid'].copy();
    #     joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)
    #
    #     # image load
    #     img = load_img(img_path)
    #
    #     joint_simdr_singles = np.zeros((self.joint_num, 3, self.output_hm_shape[0]), dtype=np.float32)
    #     joint_simdr_inters = np.zeros((self.joint_num * 2, 3, self.output_hm_shape[0]), dtype=np.float32)
    #     joint_valid_singles = np.zeros((self.joint_num), dtype=np.float32)
    #     joint_valid_inters = np.zeros((self.joint_num * 2), dtype=np.float32)
    #
    #     img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord,
    #                                                                                joint_valid,
    #                                                                                hand_type, 'test',
    #                                                                                self.joint_type)
    #     rel_root_depth = np.array(
    #                 [joint_coord[self.root_joint_idx['left'], 2] - joint_coord[self.root_joint_idx['right'], 2]],
    #                 dtype=np.float32).reshape(1)
    #      # transform to output heatmap space
    #     joint_coord, joint_valid, rel_root_depth = transform_input_to_output_space(joint_coord, joint_valid,
    #                                                                                                 rel_root_depth,
    #                                                                                                 # root_valid,
    #                                                                                                 self.root_joint_idx,
    #                                                                                                 self.joint_type)
    #     joint_simdr, joint_valid = self.generate_sa_simdr(joint_coord, joint_valid)
    #
    #     # joint_coord = self.simdr2coord(joint_simdr)
    #
    #     img = self.transform(img.astype(np.float32))
    #
    #     if hand_type == 'right' or hand_type == 'left':
    #         joint_simdr_singles = joint_simdr[self.joint_type[hand_type]]
    #         joint_valid_singles = joint_valid[self.joint_type[hand_type]]
    #     elif hand_type == 'interacting':
    #         joint_simdr_inters = joint_simdr
    #         joint_valid_inters = joint_valid
    #
    #     # print(bbox)
    #     inputs = {'img': img}
    #     targets = {'joint_simdr_singles': joint_simdr_singles, 'joint_simdr_inters': joint_simdr_inters,
    #                'rel_root_depth': rel_root_depth, 'hand_type': hand_type,
    #                'joint_simdr': joint_simdr}
    #     meta_info = {'joint_valid_singles': joint_valid_singles, 'joint_valid_inters': joint_valid_inters,
    #                 'inv_trans': inv_trans, 'joint_valid': joint_valid}
    #
    #     return inputs, targets, meta_info


    def generate_sa_simdr(self, joints, joints_valid):
        '''
        :param joints:  [num_joints, 3]
        # :param joints_vis: [num_joints, 3]
        :param joints_vis: [num_joints, 1]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        # target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight = joints_valid

        target_x = np.zeros((self.joint_num * 2,
                             self.output_hm_shape[0]),
                            dtype=np.float32)
        target_y = np.zeros((self.joint_num * 2,
                             self.output_hm_shape[1]),
                            dtype=np.float32)
        target_z = np.zeros((self.joint_num * 2,
                             self.output_hm_shape[2]),
                            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.joint_num * 2):
            # target_weight[joint_id] = \
            #     self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
            # if target_weight[joint_id] == 0:
            #     continue

            mu_x = joints[joint_id][0]
            mu_y = joints[joint_id][1]
            mu_z = joints[joint_id][2]

            x = np.arange(0, self.output_hm_shape[0], 1, np.float32)
            y = np.arange(0, self.output_hm_shape[1], 1, np.float32)
            z = np.arange(0, self.output_hm_shape[2], 1, np.float32)

            # v = target_weight[joint_id]
            # if v > 0.5:
            target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * self.sigma ** 2))) / (
                            self.sigma * np.sqrt(np.pi * 2))
            target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * self.sigma ** 2))) / (
                            self.sigma * np.sqrt(np.pi * 2))
            target_z[joint_id] = (np.exp(- ((z - mu_z) ** 2) / (2 * self.sigma ** 2))) / (
                            self.sigma * np.sqrt(np.pi * 2))
        # if self.use_different_joints_weight:
        #     target_weight = np.multiply(target_weight, self.joints_weight)

        target = np.stack((target_x, target_y, target_z), axis=1)
        return target, target_weight


    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= (self.input_img_shape[0]) or ul[1] >= self.input_img_shape[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight


    def simdr2coord(self, simdr):
        simdr_x = simdr[:, 0, :]
        simdr_y = simdr[:, 1, :]
        simdr_z = simdr[:, 2, :]
        # print(simdr_x.shape)
        idx_x = np.argmax(simdr_x, 1)
        idx_y = np.argmax(simdr_y, 1)
        idx_z = np.argmax(simdr_z, 1)
        joint_coord = np.zeros((42, 3))
        joint_coord[:, 0] = idx_x
        joint_coord[:, 1] = idx_y
        joint_coord[:, 2] = idx_z
        # print(joint_coord.shape)

        return joint_coord


    def get_dis(self, a, b, p):
        ab = b - a
        ap = p - a
        bp = p - b
        ab_L = np.linalg.norm(ab)
        r = np.dot(ap, ab) / (ab_L ** 2)
        if r > 0 and r < 1:
            dis = (np.linalg.norm(ap)) ** 2 - (r * ab_L) ** 2
            if dis < 0:
                dis = 0
            dis = math.sqrt(dis)
        elif r >= 1:
            dis = np.linalg.norm(bp)
        else:
            dis = np.linalg.norm(ap)
        return dis


    def generate_bonemap(self, joint_coord, joint_valid):
        bone_map = np.zeros((self.bone_num, 56, 56), dtype=np.float32)
        bone_valid = np.zeros((self.bone_num), dtype=np.float32)
        boneindexs = self.bone_index
        joint_coord = joint_coord[:, :2] / 8
        x = np.linspace(0, 55, 56)
        y = np.linspace(0, 55, 56)
        X, Y = np.meshgrid(x, y)
        coors = np.concatenate((X[:, :, None], Y[:, :, None]), axis=-1)
        vget_dis = np.vectorize(self.get_dis, excluded=['a', 'b'], signature='(k),(k),(k)->()')
        for b in range(0, len(boneindexs)):
            boneindex = boneindexs[b]
            if (joint_valid[[boneindex[0]]] and joint_valid[[boneindex[1]]]):
                bone_valid[b] = 1
                bone_map[b] = np.exp(- (vget_dis(coors, joint_coord[[boneindex[0]]], joint_coord[[boneindex[1]]]) ** 2) /(2 * self.sigma ** 2))
        return bone_map, bone_valid


    # def evaluate(self, preds):
    #
    #     print()
    #     print('Evaluation start...')
    #
    #     # calculate PCK parameter
    #     # max_wh = 512
    #     # threshold = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     # threshold = [0.2]
    #     # threshold = np.array(threshold)
    #
    #     gts = self.datalist
    #     pred_coord, inv_trans = preds['pred_coord'], preds['inv_trans']
    #     # print(len(pred_coord))
    #     # print(len(inv_trans))
    #
    #     # assert len(gts) == len(joint_simdr_single_out)
    #     sample_num = len(pred_coord)
    #
    #     mpjpe_rh = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_lh = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_th = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_ih = [[] for _ in range(self.joint_num * 2)]
    #
    #     mpjpe_rh_2d = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_lh_2d = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_th_2d = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_ih_2d = [[] for _ in range(self.joint_num * 2)]
    #     # # mrrpe = []
    #     # acc_hand_cls = 0;
    #     # hand_cls_cnt = 0;
    #     json_data = {'hand_type_list': [], 'bboxs': []}
    #     for n in range(sample_num):
    #         data = gts[n]
    #         cam_param, joint, gt_hand_type, hand_type_valid = data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
    #         focal = cam_param['focal']
    #         princpt = cam_param['princpt']
    #         gt_joint_coord = joint['cam_coord']
    #         joint_valid = joint['valid']
    #
    #         # restore xy coordinates to original image space
    #         pred_joint_coord_img = pred_coord[n]
    #         # print(pred_joint_coord_img.shape)
    #         # print(pred_joint_coord_img)
    #         # print()
    #         pred_joint_coord_img[:, 0] = pred_joint_coord_img[:, 0] / self.output_hm_shape[2] * self.input_img_shape[1]
    #         pred_joint_coord_img[:, 1] = pred_joint_coord_img[:, 1] / self.output_hm_shape[1] * self.input_img_shape[0]
    #         for j in range(self.joint_num * 2):
    #             pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans[n])
    #         # print(pred_joint_coord_img)
    #         # print()
    #         # for j in self.joint_type['left']:
    #         #     pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_transs[n, 1])
    #         # restore depth to original camera space
    #         pred_joint_coord_img[:, 2] = (pred_joint_coord_img[:, 2] / self.output_hm_shape[0] * 2 - 1) * (self.bbox_3d_size / 2)
    #
    #         pred_joint_coord_img[self.joint_type['right'], 2] += data['abs_depth']['right']
    #         pred_joint_coord_img[self.joint_type['left'], 2] += data['abs_depth']['left']
    #
    #         # back project to camera coordinate system
    #         pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)
    #
    #         # root joint alignment
    #         for h in ('right', 'left'):
    #             pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h], None, :]
    #             gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h], None, :]
    #
    #         # mpjpe
    #         for j in range(self.joint_num * 2):
    #             if joint_valid[j]:
    #                 # if gt_hand_type == 'right' or gt_hand_type == 'left':
    #                 #     mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
    #                 # else:
    #                 #     mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
    #                 if gt_hand_type == 'right' and data['is_two'] == False:
    #                     mpjpe_rh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
    #                     mpjpe_rh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
    #                 elif gt_hand_type == 'left' and data['is_two'] == False:
    #                     mpjpe_lh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
    #                     mpjpe_lh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
    #                 elif data['is_two'] == True:
    #                     mpjpe_th_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
    #                     mpjpe_th[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
    #                 else:
    #                     mpjpe_ih_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
    #                     mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
    #
    #         # img_path = data['img_path']
    #         # img = load_img(img_path)
    #         # vis_img = img.copy().transpose(2, 0, 1)
    #         # hand_type, bbox = data['hand_type'], data['bbox']
    #         # print(bbox)
    #     #     json_data['hand_type_list'].append(hand_type)
    #     #     json_data['bboxs'].append([bbox.tolist()])
    #     #     img = cv2.imread(img_path)
    #     #     cv2.imwrite(osp.join('./vis_img', str(n)+'.jpg'), img)
    #     # with open(osp.join('./vis_img', 'hand_type.json'), 'w') as obj:
    #     #     json.dump(json_data, obj)
    #         # gt_joint_coord = gt_joint_img_crop[n]
    #         # gt_joint_coord[:, 0] = gt_joint_coord[:, 0] / self.output_hm_shape[2] * self.input_img_shape[1]
    #         # gt_joint_coord[:, 1] = gt_joint_coord[:, 1] / self.output_hm_shape[1] * self.input_img_shape[0]
    #         # for j in self.joint_type['right']:
    #         #     gt_joint_coord[j, :2] = trans_point2d(gt_joint_coord[j, :2], inv_transs[n, 0])
    #         # for j in self.joint_type['left']:
    #         #     gt_joint_coord[j, :2] = trans_point2d(gt_joint_coord[j, :2], inv_transs[n, 1])
    #         # # restore depth to original camera space
    #         # gt_joint_coord[:, 2] = (gt_joint_coord[:, 2] / self.output_hm_shape[0] * 2 - 1) * (
    #         #             self.bbox_3d_size / 2)
    #         #
    #         # gt_coord_img = joint['img_coord']
    #         # vis_keypoints(vis_img, pred_joint_coord_img, joint_valid, self.skeleton, str(n)+'.jpg', save_path='./vis_img')
    #         # break
    #         # vis_keypoints(vis_img, gt_coord_img, joint_valid, self.skeleton, str(n) + '.jpg',save_path='./vis_img_gt')
    #
    #     # eval_summary = 'PCK for each joint: \n'
    #     # pck_the = np.zeros((threshold.shape[0], self.joint_num * 2,))
    #     # for j in range(self.joint_num * 2):
    #     #     tot_err_j = np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j])))
    #     #     # print(tot_err_j.shape)
    #     #     for index in range(threshold.shape[0]):
    #     #         pck_the[index, j] = np.mean(tot_err_j[:]/512 <= threshold[index])
    #     #         # eval_summary += (str(threshold[index]) + ': %.2f, ' %pck_the[index, j])
    #     # print(eval_summary)
    #     # pck = np.mean(pck_the, axis=1)
    #     # print(pck)
    #     #
    #     # plt.xlim(0,50)
    #     # plt.ylim(0,1)
    #     # plt.plot(threshold, pck, color='red', linestyle='-', label='botnet')
    #     # plt.title('Avg. Keypoint Error 3D')
    #     # plt.xlabel('Aligned 3D Error (mm)')
    #     # plt.ylabel('PCK')
    #     # plt.legend(loc='lower right')
    #     # plt.grid()
    #     # plt.show()
    #
    #     # 3D MPJPE
    #     tot_err = []
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in self.joint_type['right']:
    #         tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh[j]), np.stack(mpjpe_th[j]), np.stack(mpjpe_ih[j]))))
    #         # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh[j]), np.stack(mpjpe_ih[j]))))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
    #         tot_err.append(tot_err_j)
    #     for j in self.joint_type['left']:
    #         tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh[j]), np.stack(mpjpe_th[j]), np.stack(mpjpe_ih[j]))))
    #         # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh[j]), np.stack(mpjpe_ih[j]))))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
    #         tot_err.append(tot_err_j)
    #     print(eval_summary)
    #     print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
    #     print()
    #
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in self.joint_type['right']:
    #         mpjpe_rh[j] = np.mean(np.stack(mpjpe_rh[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_rh[j])
    #     print(eval_summary)
    #     print('MPJPE for right hand sequences: %.2f' % (np.mean(mpjpe_rh[:21])))
    #     print()
    #
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in self.joint_type['left']:
    #         mpjpe_lh[j] = np.mean(np.stack(mpjpe_lh[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_lh[j])
    #     print(eval_summary)
    #     print('MPJPE for left hand sequences: %.2f' % (np.mean(mpjpe_lh[21:42])))
    #     print()
    #
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in range(self.joint_num * 2):
    #         mpjpe_th[j] = np.mean(np.stack(mpjpe_th[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_th[j])
    #     print(eval_summary)
    #     print('MPJPE for two hand sequences: %.2f' % (np.mean(mpjpe_th)))
    #     print()
    #
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in range(self.joint_num * 2):
    #         mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
    #     print(eval_summary)
    #     print('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih)))
    #
    #     # 2D MPJPE
    #     tot_err = []
    #     eval_summary = 'MPJPE2d for each joint: \n'
    #     for j in self.joint_type['right']:
    #         tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh_2d[j]), np.stack(mpjpe_th_2d[j]), np.stack(mpjpe_ih_2d[j]))))
    #         # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh_2d[j]), np.stack(mpjpe_ih_2d[j]))))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
    #         tot_err.append(tot_err_j)
    #     for j in self.joint_type['left']:
    #         tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh_2d[j]), np.stack(mpjpe_th_2d[j]), np.stack(mpjpe_ih_2d[j]))))
    #         # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh_2d[j]), np.stack(mpjpe_ih_2d[j]))))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
    #         tot_err.append(tot_err_j)
    #     print(eval_summary)
    #     print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
    #     print()
    #
    #     eval_summary = 'MPJPE2d for each joint: \n'
    #     for j in self.joint_type['right']:
    #         mpjpe_rh_2d[j] = np.mean(np.stack(mpjpe_rh_2d[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_rh_2d[j])
    #     print(eval_summary)
    #     print('MPJPE for right hand sequences: %.2f' % (np.mean(mpjpe_rh_2d[:21])))
    #     print()
    #
    #     eval_summary = 'MPJPE2d for each joint: \n'
    #     for j in self.joint_type['left']:
    #         mpjpe_lh_2d[j] = np.mean(np.stack(mpjpe_lh_2d[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_lh_2d[j])
    #     print(eval_summary)
    #     print('MPJPE for left hand sequences: %.2f' % (np.mean(mpjpe_lh_2d[21:42])))
    #     print()
    #
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in range(self.joint_num * 2):
    #         mpjpe_th_2d[j] = np.mean(np.stack(mpjpe_th_2d[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_th_2d[j])
    #     print(eval_summary)
    #     print('MPJPE for two hand sequences: %.2f' % (np.mean(mpjpe_th_2d)))
    #     print()
    #
    #     eval_summary = 'MPJPE2d for each joint: \n'
    #     for j in range(self.joint_num * 2):
    #         mpjpe_ih_2d[j] = np.mean(np.stack(mpjpe_ih_2d[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih_2d[j])
    #     print(eval_summary)
    #     print('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih_2d)))

    def evaluate(self, preds):
        # for creating vedio-model input json

        print()
        print('Evaluation start...')


        # gts = self.datalist
        # pred_coord, inv_trans, target_joint_simdr_list, pred_simdrs_list, inter_attns = preds['pred_coord'], preds['inv_trans'],preds['target_joint_simdr'], preds['pred_simdrs'], preds['inter_attn']
        pred_coord, inv_trans = preds['pred_coord'], preds['inv_trans']

        # assert len(gts) == len(joint_simdr_single_out)
        sample_num = len(self.idx_list)

        mpjpe_rh = [[] for _ in range(self.joint_num * 2)]
        mpjpe_lh = [[] for _ in range(self.joint_num * 2)]
        mpjpe_th = [[] for _ in range(self.joint_num * 2)]
        mpjpe_ih = [[] for _ in range(self.joint_num * 2)]

        mpjpe_rh_2d = [[] for _ in range(self.joint_num * 2)]
        mpjpe_lh_2d = [[] for _ in range(self.joint_num * 2)]
        mpjpe_th_2d = [[] for _ in range(self.joint_num * 2)]
        mpjpe_ih_2d = [[] for _ in range(self.joint_num * 2)]

        input_json_path = osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data_input.json')
        json_data = []

        # # mrrpe = []
        # acc_hand_cls = 0;
        # hand_cls_cnt = 0;
        # for n in range(sample_num):
        n = 0
        for vedio_list in self.datalist:
            # vedio_idx, img_idx = self.idx_list[n]
            # datalist = self.datalist[vedio_idx]['datalist']
            # data = datalist[img_idx]
            vedio_path = vedio_list['vedio_path']
            data_list = vedio_list['datalist']
            json_vedio_data = []
            right_pred_joint_coord_img = None
            right_joint_valid = None
            for data in data_list:
                cam_param, joint, gt_hand_type, hand_type_valid = data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
                focal = cam_param['focal']
                princpt = cam_param['princpt']
                gt_joint_coord = joint['cam_coord']
                joint_valid = joint['valid']

                # restore xy coordinates to original image space
                pred_joint_coord_img = pred_coord[n]

                pred_joint_coord_img[:, 0] = pred_joint_coord_img[:, 0] / self.output_hm_shape[2] * self.input_img_shape[1]
                pred_joint_coord_img[:, 1] = pred_joint_coord_img[:, 1] / self.output_hm_shape[1] * self.input_img_shape[0]
                for j in range(self.joint_num * 2):
                    pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans[n])
                # restore depth to original camera space
                pred_joint_coord_img[:, 2] = (pred_joint_coord_img[:, 2] / self.output_hm_shape[0] * 2 - 1) * (self.bbox_3d_size / 2)

                pred_joint_coord_img[self.joint_type['right'], 2] += data['abs_depth']['right']
                pred_joint_coord_img[self.joint_type['left'], 2] += data['abs_depth']['left']

                # for creating vedio-model input json
                if data['is_two'] == False:
                    # json_vedio_item = {'focal': focal.tolist(), 'princpt': princpt.tolist(), 'joint_valid': joint_valid.tolist(), 'hand_type': gt_hand_type,
                    #                    'gt_joint_coord_cam': gt_joint_coord.tolist(),
                    #                    'pred_joint_coord_img': pred_joint_coord_img.tolist(), 'img_path': data['img_path'].split('/')[-1]}
                    json_vedio_item = pred_joint_coord_img.tolist()
                    json_vedio_data.append(json_vedio_item)
                    n += 1
                elif data['is_two'] and right_pred_joint_coord_img is None:
                    right_pred_joint_coord_img = pred_joint_coord_img[self.joint_type['right']].copy()
                    right_joint_valid = joint_valid[self.joint_type['right']].copy()
                    n += 1
                    continue
                else:
                    pred_joint_coord_img[self.joint_type['right']] = right_pred_joint_coord_img.copy()
                    joint_valid[self.joint_type['right']] = right_joint_valid[self.joint_type['right']].copy()
                    right_pred_joint_coord_img = None
                    right_joint_valid = None
                    gt_hand_type = 'two'

                    json_vedio_item = pred_joint_coord_img.tolist()
                    json_vedio_data.append(json_vedio_item)
                    n += 1

                # back project to camera coordinate system
                pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

                # # root joint alignment
                # for h in ('right', 'left'):
                #     pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h], None, :]
                #     gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h], None, :]
                #
                # # mpjpe
                # for j in range(self.joint_num * 2):
                #     if joint_valid[j]:
                #         if gt_hand_type == 'right':
                #             mpjpe_rh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                #             mpjpe_rh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                #         elif gt_hand_type == 'left':
                #             mpjpe_lh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                #             mpjpe_lh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                #         elif gt_hand_type == 'two':
                #             mpjpe_th_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                #             mpjpe_th[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                #         else:
                #             mpjpe_ih_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                #             mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))

            json_data.append(json_vedio_data)

                # img_path = data['img_path']
                # img = load_img(img_path)
                # vis_img = img.copy().transpose(2, 0, 1)
                # # gt_joint_coord = gt_joint_img_crop[n]
                # # gt_joint_coord[:, 0] = gt_joint_coord[:, 0] / self.output_hm_shape[2] * self.input_img_shape[1]
                # # gt_joint_coord[:, 1] = gt_joint_coord[:, 1] / self.output_hm_shape[1] * self.input_img_shape[0]
                # # for j in self.joint_type['right']:
                # #     gt_joint_coord[j, :2] = trans_point2d(gt_joint_coord[j, :2], inv_transs[n, 0])
                # # for j in self.joint_type['left']:
                # #     gt_joint_coord[j, :2] = trans_point2d(gt_joint_coord[j, :2], inv_transs[n, 1])
                # # # restore depth to original camera space
                # # gt_joint_coord[:, 2] = (gt_joint_coord[:, 2] / self.output_hm_shape[0] * 2 - 1) * (
                # #             self.bbox_3d_size / 2)
                #
                # gt_coord_img = joint['img_coord']
                # # vis_kp_bbox(vis_img, pred_joint_coord_img, joint_valid, self.skeleton, data['bbox'],str(n)+'.jpg', save_path='./vis_simdr_vedio')
                # vis_keypoints(vis_img, gt_coord_img, joint_valid, self.skeleton, str(n) + '.jpg', save_path='./vis_simdr_vedio_gt_1')
                # vis_keypoints(vis_img, pred_joint_coord_img, joint_valid, self.skeleton, str(n) + '.jpg',
                #             save_path='./vis_simdr_vedio_1')
                # target_joint_simdr = target_joint_simdr_list[n]
                # pred_simdrs = pred_simdrs_list[n]
                # #
                # vis_simdr(pred_simdrs, target_joint_simdr, str(n), './vis_simdr_1')
                # print(pred_simdrs[0,0])
                # print('-------------')
                # print(target_joint_simdr[0,0])
                # vis_attn_matrix(inter_attns[n], str(n), './vis_attn_vedio_wosoftmax')

                # n += 1
        with open(input_json_path, 'w') as obj:
            json.dump(json_data, obj)

    # def evaluate(self, preds):
    #
    #     print()
    #     print('Evaluation start...')
    #
    #     # calculate PCK parameter
    #     # max_wh = 512
    #     # threshold = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     # threshold = [0.2]
    #     # threshold = np.array(threshold)
    #
    #     # gts = self.datalist
    #     pred_coord, inv_trans = preds['pred_coord'], preds['inv_trans']
    #     # print(len(pred_coord))
    #     # print(len(inv_trans))
    #
    #     # assert len(gts) == len(joint_simdr_single_out)
    #     sample_num = len(pred_coord)
    #
    #     mpjpe_rh = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_lh = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_th = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_ih = [[] for _ in range(self.joint_num * 2)]
    #
    #     mpjpe_rh_2d = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_lh_2d = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_th_2d = [[] for _ in range(self.joint_num * 2)]
    #     mpjpe_ih_2d = [[] for _ in range(self.joint_num * 2)]
    #     # # mrrpe = []
    #     # acc_hand_cls = 0;
    #     # hand_cls_cnt = 0;
    #     json_data = {'hand_type_list': [], 'bboxs': []}
    #     for n in range(sample_num):
    #         # data = gts[n]
    #         vedio_idx, img_idx = self.idx_list[n]
    #         data = self.datalist[vedio_idx]['datalist'][img_idx]
    #
    #         cam_param, joint, gt_hand_type, hand_type_valid = data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
    #         focal = cam_param['focal']
    #         princpt = cam_param['princpt']
    #         gt_joint_coord = joint['cam_coord']
    #         joint_valid = joint['valid']
    #
    #         # restore xy coordinates to original image space
    #         pred_joint_coord_img = pred_coord[n]
    #         pred_joint_coord_img[:, 0] = pred_joint_coord_img[:, 0] / self.output_hm_shape[2] * self.input_img_shape[1]
    #         pred_joint_coord_img[:, 1] = pred_joint_coord_img[:, 1] / self.output_hm_shape[1] * self.input_img_shape[0]
    #         for j in range(self.joint_num * 2):
    #             pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans[n])
    #
    #         # restore depth to original camera space
    #         pred_joint_coord_img[:, 2] = (pred_joint_coord_img[:, 2] / self.output_hm_shape[0] * 2 - 1) * (self.bbox_3d_size / 2)
    #
    #         pred_joint_coord_img[self.joint_type['right'], 2] += data['abs_depth']['right']
    #         pred_joint_coord_img[self.joint_type['left'], 2] += data['abs_depth']['left']
    #
    #         # back project to camera coordinate system
    #         pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)
    #
    #         # root joint alignment
    #         for h in ('right', 'left'):
    #             pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h], None, :]
    #             gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h], None, :]
    #
    #         # mpjpe
    #         for j in range(self.joint_num * 2):
    #             if joint_valid[j]:
    #                 if gt_hand_type == 'right' and data['is_two'] == False:
    #                     mpjpe_rh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
    #                     mpjpe_rh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
    #                 elif gt_hand_type == 'left' and data['is_two'] == False:
    #                     mpjpe_lh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
    #                     mpjpe_lh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
    #                 elif data['is_two'] == True:
    #                     mpjpe_th_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
    #                     mpjpe_th[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
    #                 else:
    #                     mpjpe_ih_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
    #                     mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
    #
    #
    #     # 3D MPJPE
    #     tot_err = []
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in self.joint_type['right']:
    #         tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh[j]), np.stack(mpjpe_th[j]), np.stack(mpjpe_ih[j]))))
    #         # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh[j]), np.stack(mpjpe_ih[j]))))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
    #         tot_err.append(tot_err_j)
    #     for j in self.joint_type['left']:
    #         tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh[j]), np.stack(mpjpe_th[j]), np.stack(mpjpe_ih[j]))))
    #         # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh[j]), np.stack(mpjpe_ih[j]))))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
    #         tot_err.append(tot_err_j)
    #     print(eval_summary)
    #     print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
    #     print()
    #
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in self.joint_type['right']:
    #         mpjpe_rh[j] = np.mean(np.stack(mpjpe_rh[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_rh[j])
    #     print(eval_summary)
    #     print('MPJPE for right hand sequences: %.2f' % (np.mean(mpjpe_rh[:21])))
    #     print()
    #
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in self.joint_type['left']:
    #         mpjpe_lh[j] = np.mean(np.stack(mpjpe_lh[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_lh[j])
    #     print(eval_summary)
    #     print('MPJPE for left hand sequences: %.2f' % (np.mean(mpjpe_lh[21:42])))
    #     print()
    #
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in range(self.joint_num * 2):
    #         mpjpe_th[j] = np.mean(np.stack(mpjpe_th[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_th[j])
    #     print(eval_summary)
    #     print('MPJPE for two hand sequences: %.2f' % (np.mean(mpjpe_th)))
    #     print()
    #
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in range(self.joint_num * 2):
    #         mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
    #     print(eval_summary)
    #     print('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih)))
    #
    #     # 2D MPJPE
    #     tot_err = []
    #     eval_summary = 'MPJPE2d for each joint: \n'
    #     for j in self.joint_type['right']:
    #         tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh_2d[j]), np.stack(mpjpe_th_2d[j]), np.stack(mpjpe_ih_2d[j]))))
    #         # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh_2d[j]), np.stack(mpjpe_ih_2d[j]))))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
    #         tot_err.append(tot_err_j)
    #     for j in self.joint_type['left']:
    #         tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh_2d[j]), np.stack(mpjpe_th_2d[j]), np.stack(mpjpe_ih_2d[j]))))
    #         # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh_2d[j]), np.stack(mpjpe_ih_2d[j]))))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
    #         tot_err.append(tot_err_j)
    #     print(eval_summary)
    #     print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
    #     print()
    #
    #     eval_summary = 'MPJPE2d for each joint: \n'
    #     for j in self.joint_type['right']:
    #         mpjpe_rh_2d[j] = np.mean(np.stack(mpjpe_rh_2d[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_rh_2d[j])
    #     print(eval_summary)
    #     print('MPJPE for right hand sequences: %.2f' % (np.mean(mpjpe_rh_2d[:21])))
    #     print()
    #
    #     eval_summary = 'MPJPE2d for each joint: \n'
    #     for j in self.joint_type['left']:
    #         mpjpe_lh_2d[j] = np.mean(np.stack(mpjpe_lh_2d[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_lh_2d[j])
    #     print(eval_summary)
    #     print('MPJPE for left hand sequences: %.2f' % (np.mean(mpjpe_lh_2d[21:42])))
    #     print()
    #
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in range(self.joint_num * 2):
    #         mpjpe_th_2d[j] = np.mean(np.stack(mpjpe_th_2d[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_th_2d[j])
    #     print(eval_summary)
    #     print('MPJPE for two hand sequences: %.2f' % (np.mean(mpjpe_th_2d)))
    #     print()
    #
    #     eval_summary = 'MPJPE2d for each joint: \n'
    #     for j in range(self.joint_num * 2):
    #         mpjpe_ih_2d[j] = np.mean(np.stack(mpjpe_ih_2d[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih_2d[j])
    #     print(eval_summary)
    #     print('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih_2d)))
