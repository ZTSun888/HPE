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
from common.utils.bone import getbonejs


class Dataset_STB(torch.utils.data.Dataset):
    # def __init__(self, cfg, mode):
    #     self.mode = mode
    #     self.root_path = '../data/STB'
    #     self.rootnet_output_path = '../data/STB/rootnet_output/rootnet_stb_output.json'
    #     self.original_img_shape = (480, 640) # height, width
    #
    #     self.joint_num = 21 # single hand
    #     self.joint_type = {'right': np.arange(self.joint_num,self.joint_num*2), 'left': np.arange(0,self.joint_num)}
    #     self.root_joint_idx = {'right': self.joint_num, 'left': 0}
    #     self.skeleton = load_skeleton(osp.join(self.root_path, 'annotations', 'skeleton.txt'), self.joint_num*2)
    #     self.output_hm_shape = cfg.output_hm_shape
    #     self.input_img_shape = cfg.input_img_shape
    #     self.sigma = cfg.sigma
    #     self.bbox_3d_size = cfg.bbox_3d_size
    #     self.joint_shift_num = 35
    #     self.js_type = {'right': np.arange(self.joint_shift_num, self.joint_shift_num * 2),
    #                     'left': np.arange(0, self.joint_shift_num)}
    #
    #     self.datalist = []
    #     self.annot_path = osp.join(self.root_path, 'annotations', 'STB_' + self.mode + '.json')
    #     db = COCO(self.annot_path)
    #
    #     self.transform = transforms.Compose([
    #         # transforms.ToPILImage(),
    #         # transforms.Resize((self.input_img_shape[1], self.input_img_shape[0])),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #
    #     # if self.mode == 'test' and cfg.trans_test == 'rootnet':
    #     #     print("Get bbox and root depth from " + self.rootnet_output_path)
    #     #     rootnet_result = {}
    #     #     with open(self.rootnet_output_path) as f:
    #     #         annot = json.load(f)
    #     #     for i in range(len(annot)):
    #     #         rootnet_result[str(annot[i]['annot_id'])] = annot[i]
    #     # else:
    #     #     print("Get bbox and root depth from groundtruth annotation")
    #
    #     n = 0
    #     for aid in db.anns.keys():
    #         # n += 1
    #         # if n >50:
    #         #     break
    #
    #         ann = db.anns[aid]
    #         image_id = ann['image_id']
    #         img = db.loadImgs(image_id)[0]
    #
    #         seq_name = img['seq_name']
    #         img_path = osp.join(self.root_path, seq_name, img['file_name'])
    #         img_width, img_height = img['width'], img['height']
    #         cam_param = img['cam_param']
    #         focal, princpt = np.array(cam_param['focal'],dtype=np.float32), np.array(cam_param['princpt'],dtype=np.float32)
    #
    #         joint_img = np.array(ann['joint_img'],dtype=np.float32)
    #         joint_cam = np.array(ann['joint_cam'],dtype=np.float32)
    #         joint_valid = np.array(ann['joint_valid'],dtype=np.float32)
    #
    #         # transform single hand data to double hand data structure
    #         hand_type = ann['hand_type']
    #         joint_img_dh = np.zeros((self.joint_num*2,2),dtype=np.float32)
    #         joint_cam_dh = np.zeros((self.joint_num*2,3),dtype=np.float32)
    #         joint_valid_dh = np.zeros((self.joint_num*2),dtype=np.float32)
    #         joint_img_dh[self.joint_type[hand_type]] = joint_img
    #         joint_cam_dh[self.joint_type[hand_type]] = joint_cam
    #         joint_valid_dh[self.joint_type[hand_type]] = joint_valid
    #         joint_img = joint_img_dh; joint_cam = joint_cam_dh; joint_valid = joint_valid_dh;
    #
    #         # if self.mode == 'test' and cfg.trans_test == 'rootnet':
    #         #     bbox = np.array(rootnet_result[str(aid)]['bbox'],dtype=np.float32)
    #         #     abs_depth = rootnet_result[str(aid)]['abs_depth']
    #         # else:
    #         bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
    #         bbox = process_bbox(bbox, (img_height, img_width))
    #         abs_depth = joint_cam[self.root_joint_idx[hand_type], 2]
    #
    #
    #         cam_param = {'focal': focal, 'princpt': princpt}
    #         joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
    #         data = {'img_path': img_path, 'bbox': bbox, 'cam_param': cam_param, 'joint': joint, 'hand_type': hand_type, 'abs_depth': abs_depth}
    #         self.datalist.append(data)
    #
    #     print('Number of annotations in hand sequences: ' + str(len(self.datalist)))

    def __init__(self, cfg, mode):
        # for creating vedio-model input json
        self.mode = 'train'
        self.root_path = '../data/STB'
        self.original_img_shape = (480, 640)  # height, width

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

                # transform single hand data to double hand data structure
                hand_type = ann['hand_type']
                joint_img_dh = np.zeros((self.joint_num * 2, 2), dtype=np.float32)
                joint_cam_dh = np.zeros((self.joint_num * 2, 3), dtype=np.float32)
                joint_valid_dh = np.zeros((self.joint_num * 2), dtype=np.float32)
                joint_img_dh[self.joint_type[hand_type]] = joint_img
                joint_cam_dh[self.joint_type[hand_type]] = joint_cam
                joint_valid_dh[self.joint_type[hand_type]] = joint_valid
                joint_img = joint_img_dh;
                joint_cam = joint_cam_dh;
                joint_valid = joint_valid_dh;

                bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))
                abs_depth = joint_cam[self.root_joint_idx[hand_type], 2]

                cam_param = {'focal': focal, 'princpt': princpt}
                joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
                data = {'img_path': img_path, 'bbox': bbox, 'cam_param': cam_param, 'joint': joint, 'hand_type': hand_type,
                        'abs_depth': abs_depth}
                datalist.append(data)
                self.idx_list.append([vedio_idx, img_idx])
                img_idx += 1

            vedio_list = {'vedio_path': vedio_path, 'datalist': datalist}
            self.datalist.append(vedio_list)
            vedio_idx += 1

        print('Number of annotations in hand sequences: ' + str(len(self.idx_list)))

    # def __len__(self):
    #     return len(self.datalist)
    def __len__(self):
    #     for creating vedio-model input json
        return len(self.idx_list)

    def __getitem__(self, idx):
        # for creating vedio-model input json
        vedio_idx, img_idx = self.idx_list[idx]
        data = self.datalist[vedio_idx]['datalist'][img_idx]
        img_path, bbox, joint, hand_type = data['img_path'], data['bbox'], data['joint'], data['hand_type']
        joint_cam = joint['cam_coord'].copy(); joint_img = joint['img_coord'].copy(); joint_valid = joint['valid'].copy();
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1)

        # image load
        img = load_img(img_path)
        # augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type, 'test', self.joint_type)
        img = self.transform(img.astype(np.float32))
        rel_root_depth = np.zeros((1),dtype=np.float32)

        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth = transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, self.root_joint_idx, self.joint_type)

        joint_coord_single = joint_coord[self.joint_type[hand_type]]
        joint_valid_single = joint_valid[self.joint_type[hand_type]]

        joint_simdr_singles, joint_valid_singles = self.generate_sa_simdr(joint_coord_single, joint_valid_single)

        inputs = {'img': img}
        targets = {'joint_simdr_singles': joint_simdr_singles, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}#, 'js_single': js_single}
        meta_info = {'joint_valid_singles': joint_valid_singles, 'inv_trans': inv_trans, 'hand_type_valid': 1}#, 'js_valid_singles': js_valid_single}
        return inputs, targets, meta_info


    # def __getitem__(self, idx):
    #     data = self.datalist[idx]
    #     img_path, bbox, joint, hand_type = data['img_path'], data['bbox'], data['joint'], data['hand_type']
    #     joint_cam = joint['cam_coord'].copy(); joint_img = joint['img_coord'].copy(); joint_valid = joint['valid'].copy();
    #     joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1)
    #
    #     # image load
    #     img = load_img(img_path)
    #     # augmentation
    #     img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type, self.mode, self.joint_type)
    #     img = self.transform(img.astype(np.float32))
    #     rel_root_depth = np.zeros((1),dtype=np.float32)
    #
    #     # transform to output heatmap space
    #     joint_coord, joint_valid, rel_root_depth = transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, self.root_joint_idx, self.joint_type)
    #
    #
    #     joint_coord_single = joint_coord[self.joint_type[hand_type]]
    #     joint_valid_single = joint_valid[self.joint_type[hand_type]]
    #     # js_single, js_valid_single = getbonejs(joint_coord_single, joint_valid_single)
    #
    #     joint_simdr_singles, joint_valid_singles = self.generate_sa_simdr(joint_coord_single, joint_valid_single)
    #
    #     inputs = {'img': img}
    #     targets = {'joint_simdr_singles': joint_simdr_singles, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}#, 'js_single': js_single}
    #     meta_info = {'joint_valid_singles': joint_valid_singles, 'inv_trans': inv_trans, 'hand_type_valid': 1}#, 'js_valid_singles': js_valid_single}
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

        target_x = np.zeros((self.joint_num,
                             self.output_hm_shape[0]),
                            dtype=np.float32)
        target_y = np.zeros((self.joint_num,
                             self.output_hm_shape[1]),
                            dtype=np.float32)
        target_z = np.zeros((self.joint_num,
                             self.output_hm_shape[2]),
                            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.joint_num):
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
        joint_coord = np.zeros((21, 3))
        joint_coord[:, 0] = idx_x
        joint_coord[:, 1] = idx_y
        joint_coord[:, 2] = idx_z
        # print(joint_coord.shape)

        return joint_coord


    # def evaluate(self, preds):
    #
    #     print()
    #     print('Evaluation start...')
    #
    #     gts = self.datalist
    #     preds_joint_coord, inv_trans = preds['pred_coord'], preds['inv_trans']
    #     assert len(gts) == len(preds_joint_coord)
    #     sample_num = len(gts)
    #
    #     mpjpe = [[] for _ in range(self.joint_num)] # treat right and left hand identical
    #     mpjpe_2d = [[] for _ in range(self.joint_num)]
    #     acc_hand_cls = 0
    #     for n in range(sample_num):
    #         data = gts[n]
    #         bbox, cam_param, joint, gt_hand_type = data['bbox'], data['cam_param'], data['joint'], data['hand_type']
    #         focal = cam_param['focal']
    #         princpt = cam_param['princpt']
    #         gt_joint_coord_dh = joint['cam_coord']
    #         joint_valid_dh = joint['valid']
    #         gt_joint_coord = gt_joint_coord_dh[self.joint_type[gt_hand_type]].copy()
    #         joint_valid = joint_valid_dh[self.joint_type[gt_hand_type]].copy()
    #
    #         # restore coordinates to original space
    #         pred_joint_coord_img = preds_joint_coord[n].copy()
    #         pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/self.output_hm_shape[2]*self.input_img_shape[1]
    #         pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/self.output_hm_shape[1]*self.input_img_shape[0]
    #         for j in range(self.joint_num):
    #             pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2],inv_trans[n])
    #         pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/self.output_hm_shape[0] * 2 - 1) * (self.bbox_3d_size/2)
    #         pred_joint_coord_img[:,2] = pred_joint_coord_img[:,2] + data['abs_depth']
    #
    #         # back project to camera coordinate system
    #         pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)
    #
    #         # root joint alignment
    #         pred_joint_coord_cam[:] = pred_joint_coord_cam[:] - pred_joint_coord_cam[0,None,:]
    #         # pred_joint_coord_cam[self.joint_type['left']] = pred_joint_coord_cam[self.joint_type['left']] - pred_joint_coord_cam[self.root_joint_idx['left'],None,:]
    #         gt_joint_coord[:] = gt_joint_coord[:] - gt_joint_coord[0,None,:]
    #         # gt_joint_coord[self.joint_type['left']] = gt_joint_coord[self.joint_type['left']] - gt_joint_coord[self.root_joint_idx['left'],None,:]
    #
    #         # select right or left hand using groundtruth hand type
    #         pred_joint_coord_cam = pred_joint_coord_cam
    #         # gt_joint_coord = gt_joint_coord[self.joint_type[gt_hand_type]]
    #         # joint_valid = joint_valid[self.joint_type[gt_hand_type]]
    #
    #         # mpjpe save
    #         for j in range(self.joint_num):
    #             if joint_valid[j]:
    #                 mpjpe[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
    #                 mpjpe_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2])**2)))
    #
    #         # img_path = data['img_path']
    #         # img = load_img(img_path)
    #         # vis_img = img.copy().transpose(2, 0, 1)
    #         # # # hand_type, bbox = data['hand_type'], data['bbox']
    #         # pred_joint_coord_img_dh = np.zeros((42, 3), dtype=np.float32)
    #         # pred_joint_coord_img_dh[self.joint_type[gt_hand_type]] = pred_joint_coord_img
    #         # bboxs = []
    #         # bboxs.append(bbox)
    #         # # # gt_joint_coord_img_dh = joint['img_coord']
    #         # vis_kp_bbox(vis_img, pred_joint_coord_img_dh, joint_valid_dh, self.skeleton, bboxs, str(n)+'.jpg', save_path='./vis_img_STB')
    #
    #     # print('Handedness accuracy: ' + str(acc_hand_cls / sample_num))
    #
    #     eval_summary = 'MPJPE for each joint: \n'
    #     for j in range(self.joint_num):
    #         mpjpe[j] = np.mean(np.stack(mpjpe[j]))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % mpjpe[j])
    #     print(eval_summary)
    #     print('MPJPE: %.2f' % (np.mean(mpjpe)))
    #
    #     tot_err = []
    #     eval_summary = 'MPJPE2d for each joint: \n'
    #     for j in range(self.joint_num):
    #         tot_err_j = np.mean(np.stack(mpjpe_2d[j]))
    #         # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh_2d[j]), np.stack(mpjpe_ih_2d[j]))))
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
    #         tot_err.append(tot_err_j)
    #     print(eval_summary)
    #     print('MPJPE2d for all hand sequences: %.2f' % (np.mean(tot_err)))
    #     print()

    def evaluate(self, preds):
        # for creating vedio-model input json

        print()
        print('Evaluation start...')

        pred_coord, inv_trans = preds['pred_coord'], preds['inv_trans']

        input_json_path = osp.join(self.annot_path, 'STB_' + self.mode + '_data_input.json')
        json_data = []

        n = 0
        for vedio_list in self.datalist:

            data_list = vedio_list['datalist']
            json_vedio_data = []
            for data in data_list:
                # restore xy coordinates to original image space
                pred_joint_coord_img = pred_coord[n]

                pred_joint_coord_img[:, 0] = pred_joint_coord_img[:, 0] / self.output_hm_shape[2] * self.input_img_shape[1]
                pred_joint_coord_img[:, 1] = pred_joint_coord_img[:, 1] / self.output_hm_shape[1] * self.input_img_shape[0]
                for j in range(self.joint_num):
                    pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans[n])
                # restore depth to original camera space
                pred_joint_coord_img[:, 2] = (pred_joint_coord_img[:, 2] / self.output_hm_shape[0] * 2 - 1) * (self.bbox_3d_size / 2)

                pred_joint_coord_img[:, 2] = pred_joint_coord_img[:, 2] + data['abs_depth']
                # pred_joint_coord_img[self.joint_type['left'], 2] += data['abs_depth']['left']

                # for creating vedio-model input json
                json_vedio_item = pred_joint_coord_img.tolist()
                json_vedio_data.append(json_vedio_item)
                n += 1

            json_data.append(json_vedio_data)

        with open(input_json_path, 'w') as obj:
            json.dump(json_data, obj)