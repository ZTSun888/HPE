import numpy as np
import torch
import torch.utils.data
import cv2
import os
import os.path as osp
from torchvision import transforms
from common.utils.preprocessing import load_img, load_skeleton, process_bbox, get_aug_config, augmentation, transform_input_to_output_space, generate_patch_image, trans_point2d
from common.utils.transforms import world2cam, cam2pixel, pixel2cam
from common.utils.vis import vis_keypoints, vis_3d_keypoints, vis_kp_bbox
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO
import sys
sys.path.append('..')
from main.config import cfg


class DatasetVedioSTB(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, pad, inference=False):
        self.mode = mode
        self.root_path = '../data/STB'
        self.original_img_shape = (480, 640)  # height, width

        self.pad = pad
        self.inference = inference
        if self.inference:
            self.opt_coord_list = []

        self.joint_num = 21  # single hand
        self.joint_type = {'right': np.arange(self.joint_num, self.joint_num * 2), 'left': np.arange(0, self.joint_num)}
        self.root_joint_idx = {'right': self.joint_num, 'left': 0}
        self.skeleton = load_skeleton(osp.join(self.root_path, 'annotations', 'skeleton.txt'), self.joint_num * 2)
        self.output_hm_shape = cfg.output_hm_shape
        self.input_img_shape = cfg.input_img_shape
        self.sigma = cfg.sigma
        self.bbox_3d_size = cfg.bbox_3d_size
        self.joint_shift_num = 35
        self.js_type = {'right': np.arange(self.joint_shift_num, self.joint_shift_num * 2),
                        'left': np.arange(0, self.joint_shift_num)}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datalist = []
        self.annot_path = osp.join(self.root_path, 'vedio_annot', self.mode)
        vedio_idx = 0
        self.idx_list = []

        # with open(osp.join(self.root_path, 'annotations', 'STB_' + self.mode + '_data_input.json')) as f:
        #     pred_dbs = json.load(f)
        # for InterHand model predictions as input
        with open(osp.join(self.root_path, 'annotations', 'STB_' + self.mode + '_data_input_inter.json')) as f:
            pred_dbs = json.load(f)

        for file_name in os.listdir(self.annot_path):
            db = COCO(os.path.join(self.annot_path, file_name))
            vedio_path = file_name
            datalist = []
            img_idx = 0
            for aid in db.anns.keys():

                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]

                seq_name = img['seq_name']
                img_path = osp.join(self.root_path, seq_name, img['file_name'])
                img_width, img_height = img['width'], img['height']
                cam_param = img['cam_param']
                focal, princpt = np.array(cam_param['focal'], dtype=np.float32), np.array(cam_param['princpt'],
                                                                                          dtype=np.float32)

                joint_img = np.array(ann['joint_img'], dtype=np.float32)
                joint_cam = np.array(ann['joint_cam'], dtype=np.float32)
                joint_valid = np.array(ann['joint_valid'], dtype=np.float32)

                pred_joint_coord_img = np.array(pred_dbs[vedio_idx][img_idx], dtype=np.float32)

                # transform single hand data to double hand data structure
                hand_type = ann['hand_type']
                joint_img_dh = np.zeros((self.joint_num * 2, 2), dtype=np.float32)
                joint_cam_dh = np.zeros((self.joint_num * 2, 3), dtype=np.float32)
                joint_valid_dh = np.zeros((self.joint_num * 2), dtype=np.float32)
                joint_img_dh[self.joint_type[hand_type]] = joint_img
                joint_cam_dh[self.joint_type[hand_type]] = joint_cam
                # joint_valid_dh[self.joint_type[hand_type]] = joint_valid
                joint_img = joint_img_dh;
                joint_cam = joint_cam_dh;
                # joint_valid = joint_valid_dh;

                bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))

                input_root_coord = pred_joint_coord_img[self.root_joint_idx[hand_type]]

                cam_param = {'focal': focal, 'princpt': princpt}
                joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid, 'pred_joint_coord_img': pred_joint_coord_img}
                data = {'img_path': img_path, 'bbox': bbox, 'cam_param': cam_param, 'joint': joint, 'hand_type': hand_type,
                        'img_width': img_width, 'img_height': img_height, 'input_root_coord': input_root_coord,}
                datalist.append(data)
                self.idx_list.append([vedio_idx, img_idx])
                img_idx += 1

            vedio_list = {'vedio_path': vedio_path, 'datalist': datalist}
            self.datalist.append(vedio_list)
            vedio_idx += 1

        print('Number of annotations in hand sequences: ' + str(len(self.idx_list)))

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        vedio_idx, img_idx = self.idx_list[idx]
        data = self.datalist[vedio_idx]['datalist'][img_idx]
        img_width, img_height = data['img_width'], data['img_height']

        input_list = []
        joint_valid_list = []

        for i in range(img_idx - self.pad + 1, img_idx + 1):
            if i < 0:
                data = self.datalist[vedio_idx]['datalist'][0]
            else:
                data = self.datalist[vedio_idx]['datalist'][i]
            joint = data['joint']
            input_list.append(joint['pred_joint_coord_img'].copy())
            joint_valid_list.append(joint['valid'].copy())


        input_list = np.array(input_list, dtype=np.float32)
        for i in range(input_list.shape[0]):
            # for h in ('right', 'left'):
            input_list[i, :, 2] = input_list[i, :, 2] - input_list[i, 0, 2]

        joint_valid_list = np.array(joint_valid_list, dtype=np.float32)

        joint_valid = joint_valid_list[-1, :]


        # norm to [-1, 1]
        input_list[:, :, 0] = (input_list[:, :, 0] - (img_width / 2)) / (img_width / 2)
        input_list[:, :, 1] = (input_list[:, :, 1] - (img_height / 2)) / (img_height / 2)
        input_list[:, :, 2] = input_list[:, :, 2] / 200

        inputs = input_list

        meta_info = {'joint_valid': joint_valid, 'vedio_idx': vedio_idx, 'bone_valid': self.get_bone_valid(joint_valid)}
        return inputs, meta_info

    def get_bone_valid(self, joint_valid):
        bone_indexs = cfg.bone_index
        bone_valid = np.zeros(cfg.bone_num * 2)
        for i in range(len(bone_indexs)):
            bone_valid[i] = joint_valid[bone_indexs[i][0]] * joint_valid[bone_indexs[i][1]]
        return bone_valid

    def cal_acc(self, trail_opt):

        accel_opt = trail_opt[:-2] - 2 * trail_opt[1:-1] + trail_opt[2:]
        normed_opt = np.linalg.norm(accel_opt, axis=2)
        new_vis_opt = np.ones(len(normed_opt), dtype=bool)
        acc_opt = np.mean(np.mean(normed_opt[new_vis_opt], axis=1))

        return acc_opt

    def evaluate(self, preds):

        print()
        print('Evaluation start...')

        out = preds['pred_coord']
        assert len(self.idx_list) == len(out)

        mpjpe = [[] for _ in range(self.joint_num)] # treat right and left hand identical
        mpjpe_2d = [[] for _ in range(self.joint_num)]
        n = 0

        acc_list = []
        acc_gt_list = []

        for vedio_list in self.datalist:
            data_list = vedio_list['datalist']
            video_len = len(data_list)
            video_hand_type = data_list[0]['hand_type']

            opt_trail = np.zeros((video_len, 21, 3), dtype=np.float32)
            img_trail = np.zeros((video_len, 21, 3), dtype=np.float32)

            coord_idx = 0
            for data in data_list:
                cam_param, joint, gt_hand_type= data['cam_param'], data['joint'], data['hand_type']
                focal = cam_param['focal']
                princpt = cam_param['princpt']
                gt_joint_coord_dh = joint['cam_coord']
                joint_valid_dh = joint['valid']
                gt_joint_coord = gt_joint_coord_dh[self.joint_type[gt_hand_type]].copy()
                joint_valid = joint_valid_dh[self.joint_type[gt_hand_type]].copy()

                # restore coordinates to original space
                img_width, img_height = data['img_width'], data['img_height']

                pred_joint_coord_img = out[n]
                input_joint_coord = joint['pred_joint_coord_img']

                pred_joint_coord_img_input = data['input_root_coord']
                pred_joint_coord_img[:, 0] = (pred_joint_coord_img[:, 0] * (img_width / 2)) + (img_width / 2)
                pred_joint_coord_img[:, 1] = (pred_joint_coord_img[:, 1] * (img_height / 2)) + (img_height / 2)
                pred_joint_coord_img[:, 2] = pred_joint_coord_img[:, 2] * 200
                pred_joint_coord_img[:, 2] = pred_joint_coord_img[: , 2] + pred_joint_coord_img_input[2]

                # back project to camera coordinate system
                pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)
                input_joint_coord_cam = pixel2cam(input_joint_coord, focal, princpt).copy()

                # root joint alignment
                pred_joint_coord_cam[:] = pred_joint_coord_cam[:] - pred_joint_coord_cam[0,None,:]
                gt_joint_coord[:] = gt_joint_coord[:] - gt_joint_coord[0,None,:]

                # gt_joint_coord = gt_joint_coord[self.joint_type[gt_hand_type]]
                # joint_valid = joint_valid[self.joint_type[gt_hand_type]]

                # mpjpe save
                for j in range(self.joint_num):
                    if joint_valid[j]:
                        mpjpe[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                        mpjpe_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2])**2)))


                opt_trail[coord_idx] = pred_joint_coord_cam
                img_trail[coord_idx] = input_joint_coord_cam

                n += 1

            acc_list.append(self.cal_acc(opt_trail))
            acc_gt_list.append(self.cal_acc(img_trail))
            # print(acc_list[0])
            # video_idx += 1

            # img_path = data['img_path']
            # img = load_img(img_path)
            # vis_img = img.copy().transpose(2, 0, 1)
            # # # hand_type, bbox = data['hand_type'], data['bbox']
            # pred_joint_coord_img_dh = np.zeros((42, 3), dtype=np.float32)
            # pred_joint_coord_img_dh[self.joint_type[gt_hand_type]] = pred_joint_coord_img
            # bboxs = []
            # bboxs.append(bbox)
            # # # gt_joint_coord_img_dh = joint['img_coord']
            # vis_kp_bbox(vis_img, pred_joint_coord_img_dh, joint_valid_dh, self.skeleton, bboxs, str(n)+'.jpg', save_path='./vis_img_STB')

        # print('Handedness accuracy: ' + str(acc_hand_cls / sample_num))

        acc = np.mean(np.array(acc_list))
        acc_gt = np.mean(np.array(acc_gt_list))
        print('acc:' + str(acc))
        print('acc_gt:' + str(acc_gt))

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num):
            mpjpe[j] = np.mean(np.stack(mpjpe[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe[j])
        print(eval_summary)
        print('MPJPE: %.2f' % (np.mean(mpjpe)))

        tot_err = []
        eval_summary = 'MPJPE2d for each joint: \n'
        for j in range(self.joint_num):
            tot_err_j = np.mean(np.stack(mpjpe_2d[j]))
            # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh_2d[j]), np.stack(mpjpe_ih_2d[j]))))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        print(eval_summary)
        print('MPJPE2d for all hand sequences: %.2f' % (np.mean(tot_err)))
        print()

