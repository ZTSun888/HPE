import numpy as np
import torch
import torch.utils.data
import cv2
import os
import os.path as osp
from torchvision import transforms
from common.utils.preprocessing import load_img, load_skeleton, process_bbox, get_aug_config, augmentation, transform_input_to_output_space, generate_patch_image, trans_point2d
# from common.utils.transforms import world2cam, cam2pixel, pixel2cam
from common.utils.vis import vis_keypoints, vis_3d_keypoints, vis_kp_bbox
from PIL import Image, ImageDraw
import random
import json
import math
from common.utils.bone import getbonejs

def cam2pixel(cam_coord):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * 375.864 + 270
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * 375.864 + 480
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord):
    x = (pixel_coord[:, 0] - 270) / 375.864 * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - 480) / 375.864 * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord


class Dataset_Ego3D(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.mode = mode
        self.root_path = '../data/Ego3DHands (static)'
        # self.rootnet_output_path = '../data/STB/rootnet_output/rootnet_stb_output.json'
        # self.original_img_shape = (480, 640) # height, width

        self.joint_num = 21  # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num * 2)}
        self.skeleton = load_skeleton(osp.join(self.root_path, 'skeleton.txt'), self.joint_num * 2)

        self.output_hm_shape = cfg.output_hm_shape
        self.input_img_shape = cfg.input_img_shape
        self.sigma = cfg.sigma
        self.bbox_3d_size = cfg.bbox_3d_size

        self.datalist = []
        self.annot_path = osp.join(self.root_path, mode, 'Ego3DHands_static_' + self.mode + '_data.json')
        with open(self.annot_path) as f:
            json_data_list = json.load(f)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datalist_rh = []
        self.datalist_lh = []
        self.datalist_ih = []
        self.datalist_th = []

        n = 0
        for json_data in json_data_list:
            # n += 1
            # if n >50:
            #     break

            data_idx = json_data['img_idx']
            img_path = os.path.join(self.root_path, self.mode, data_idx, 'color_new.png')
            hand_type = json_data['hand_type']
            joint_img = np.array(json_data['joint_img'], dtype=np.float32)
            bboxs = np.array(json_data['bboxs'], dtype=np.float32)
            img_shape = json_data['shape']
            joint_valid = np.zeros((self.joint_num * 2))

            # joint = {}
            data = {'img_path': img_path, 'hand_type': hand_type,
                    'img_coord': joint_img, 'is_two': False}

            if hand_type == 'right':
                # rh += 1
                # if (rh > 50):
                #     continue
                bboxs[0] = process_bbox(bboxs[0], img_shape)
                data['bbox'] = bboxs[0]
                joint_valid[:self.joint_num] = 1
                data['valid'] = joint_valid
                self.datalist_rh.append(data)
            elif hand_type == 'left':
                # lh += 1
                # if (lh > 50):
                #     continue
                bboxs[0] = process_bbox(bboxs[0], img_shape)
                data['bbox'] = bboxs[0]
                joint_valid[self.joint_num:] = 1
                data['valid'] = joint_valid
                self.datalist_lh.append(data)
            elif hand_type == 'two':
                # continue
                # th += 1
                # if (th > 50):
                #     continue
                # self.datalist_th.append(data)

                joint_valid[:self.joint_num] = 1
                bboxs[0] = process_bbox(bboxs[0], img_shape)
                data_right = {'img_path': img_path, 'bbox': bboxs[0],
                              'img_coord': joint_img, 'valid': joint_valid,
                              'hand_type': 'right'}
                # print(data_right['joint']['valid'])

                joint_valid[self.joint_num:] = 1
                bboxs[1] = process_bbox(bboxs[1], img_shape)
                data_left = {'img_path': img_path, 'bbox': bboxs[1],
                              'img_coord': joint_img, 'valid': joint_valid,
                              'hand_type': 'right'}
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
                bboxs[0] = process_bbox(bboxs[0], img_shape)
                data['bbox'] = bboxs[0]
                joint_valid[:] = 1
                data['valid'] = joint_valid
                self.datalist_ih.append(data)
        self.datalist = self.datalist_rh + self.datalist_lh + self.datalist_th + self.datalist_ih
        # self.datalist = self.datalist_rh + self.datalist_lh + self.datalist_th
        # self.datalist = self.datalist_ih
        print('Number of annotations in right hand sequences: ' + str(len(self.datalist_rh)))
        print('Number of annotations in left hand sequences: ' + str(len(self.datalist_lh)))
        print('Number of annotations in two hand sequences: ' + str(len(self.datalist_th)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))

    # def __init__(self, cfg, mode):
    #     # for creating vedio-model input json
    #     self.mode = 'train'
    #     self.root_path = '../data/STB'
    #     self.original_img_shape = (480, 640)  # height, width
    #
    #     self.joint_num = 21  # single hand
    #     self.joint_type = {'right': np.arange(self.joint_num, self.joint_num * 2), 'left': np.arange(0, self.joint_num)}
    #     self.root_joint_idx = {'right': self.joint_num, 'left': 0}
    #     self.skeleton = load_skeleton(osp.join(self.root_path, 'annotations', 'skeleton.txt'), self.joint_num * 2)
    #     self.output_hm_shape = cfg.output_hm_shape
    #     self.input_img_shape = cfg.input_img_shape
    #     self.sigma = cfg.sigma
    #     self.bbox_3d_size = cfg.bbox_3d_size
    #     self.joint_shift_num = 35
    #     self.js_type = {'right': np.arange(self.joint_shift_num, self.joint_shift_num * 2),
    #                     'left': np.arange(0, self.joint_shift_num)}
    #     self.transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #
    #     self.datalist = []
    #     self.annot_path = osp.join(self.root_path, 'vedio_annot', self.mode)
    #     vedio_idx = 0
    #     self.idx_list = []
    #
    #     for file_name in os.listdir(self.annot_path):
    #         db = COCO(os.path.join(self.annot_path, file_name))
    #         vedio_path = file_name
    #         datalist = []
    #         img_idx = 0
    #         for aid in db.anns.keys():
    #
    #             ann = db.anns[aid]
    #             image_id = ann['image_id']
    #             img = db.loadImgs(image_id)[0]
    #
    #             seq_name = img['seq_name']
    #             img_path = osp.join(self.root_path, seq_name, img['file_name'])
    #             img_width, img_height = img['width'], img['height']
    #             cam_param = img['cam_param']
    #             focal, princpt = np.array(cam_param['focal'], dtype=np.float32), np.array(cam_param['princpt'],
    #                                                                                       dtype=np.float32)
    #
    #             joint_img = np.array(ann['joint_img'], dtype=np.float32)
    #             joint_cam = np.array(ann['joint_cam'], dtype=np.float32)
    #             joint_valid = np.array(ann['joint_valid'], dtype=np.float32)
    #
    #             # transform single hand data to double hand data structure
    #             hand_type = ann['hand_type']
    #             joint_img_dh = np.zeros((self.joint_num * 2, 2), dtype=np.float32)
    #             joint_cam_dh = np.zeros((self.joint_num * 2, 3), dtype=np.float32)
    #             joint_valid_dh = np.zeros((self.joint_num * 2), dtype=np.float32)
    #             joint_img_dh[self.joint_type[hand_type]] = joint_img
    #             joint_cam_dh[self.joint_type[hand_type]] = joint_cam
    #             joint_valid_dh[self.joint_type[hand_type]] = joint_valid
    #             joint_img = joint_img_dh;
    #             joint_cam = joint_cam_dh;
    #             joint_valid = joint_valid_dh;
    #
    #             bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
    #             bbox = process_bbox(bbox, (img_height, img_width))
    #             abs_depth = joint_cam[self.root_joint_idx[hand_type], 2]
    #
    #             cam_param = {'focal': focal, 'princpt': princpt}
    #             joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
    #             data = {'img_path': img_path, 'bbox': bbox, 'cam_param': cam_param, 'joint': joint, 'hand_type': hand_type,
    #                     'abs_depth': abs_depth}
    #             datalist.append(data)
    #             self.idx_list.append([vedio_idx, img_idx])
    #             img_idx += 1
    #
    #         vedio_list = {'vedio_path': vedio_path, 'datalist': datalist}
    #         self.datalist.append(vedio_list)
    #         vedio_idx += 1
    #
    #     print('Number of annotations in hand sequences: ' + str(len(self.idx_list)))

    def __len__(self):
        return len(self.datalist)
    # def __len__(self):
    #     for creating vedio-model input json
    #     return len(self.idx_list)

    # def __getitem__(self, idx):
    #     # for creating vedio-model input json
    #     vedio_idx, img_idx = self.idx_list[idx]
    #     data = self.datalist[vedio_idx]['datalist'][img_idx]
    #     img_path, bbox, joint, hand_type = data['img_path'], data['bbox'], data['joint'], data['hand_type']
    #     joint_cam = joint['cam_coord'].copy(); joint_img = joint['img_coord'].copy(); joint_valid = joint['valid'].copy();
    #     joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1)
    #
    #     # image load
    #     img = load_img(img_path)
    #     # augmentation
    #     img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type, 'test', self.joint_type)
    #     img = self.transform(img.astype(np.float32))
    #     rel_root_depth = np.zeros((1),dtype=np.float32)
    #
    #     # transform to output heatmap space
    #     joint_coord, joint_valid, rel_root_depth = transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, self.root_joint_idx, self.joint_type)
    #
    #     joint_coord_single = joint_coord[self.joint_type[hand_type]]
    #     joint_valid_single = joint_valid[self.joint_type[hand_type]]
    #
    #     joint_simdr_singles, joint_valid_singles = self.generate_sa_simdr(joint_coord_single, joint_valid_single)
    #
    #     inputs = {'img': img}
    #     targets = {'joint_simdr_singles': joint_simdr_singles, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}#, 'js_single': js_single}
    #     meta_info = {'joint_valid_singles': joint_valid_singles, 'inv_trans': inv_trans, 'hand_type_valid': 1}#, 'js_valid_singles': js_valid_single}
    #     return inputs, targets, meta_info


    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, hand_type = data['img_path'], data['bbox'], data['hand_type']

        joint_coord = data['img_coord'].copy()
        joint_valid = data['valid'].copy()

        # image load
        img = load_img(img_path)

        # vis_img = load_img(img_path).transpose(2, 0, 1)
        # # print(joint_coord.shape)
        # bboxs = [bbox]
        # vis_kp_bbox(vis_img, joint_coord, joint_valid, self.skeleton, bboxs, str(idx) + '_bboxs.jpg', save_path='.')
        joint_simdr_singles = np.zeros((self.joint_num, 3, self.output_hm_shape[0]), dtype=np.float32)
        joint_simdr_inters = np.zeros((self.joint_num * 2, 3, self.output_hm_shape[0]), dtype=np.float32)
        joint_valid_singles = np.zeros((self.joint_num), dtype=np.float32)
        joint_valid_inters = np.zeros((self.joint_num * 2), dtype=np.float32)
        # augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type, self.mode, self.joint_type)
        img = self.transform(img.astype(np.float32))
        rel_root_depth = np.zeros((1),dtype=np.float32)

        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth = transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, self.root_joint_idx, self.joint_type)
        joint_simdr, joint_valid = self.generate_sa_simdr(joint_coord, joint_valid)
        if hand_type == 'right' or hand_type == 'left':
            joint_simdr_singles = joint_simdr[self.joint_type[hand_type]]
            joint_valid_singles = joint_valid[self.joint_type[hand_type]]

            joint_valid_inters[self.joint_type[hand_type]] = joint_valid_singles
            joint_simdr_inters[self.joint_type[hand_type]] = joint_simdr_singles
        elif hand_type == 'interacting':
            joint_simdr_inters = joint_simdr
            joint_valid_inters = joint_valid

        inputs = {'img': img}
        targets = {'joint_simdr_singles': joint_simdr_singles, 'joint_simdr_inters': joint_simdr_inters,
                   'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
        meta_info = {'joint_valid_singles': joint_valid_singles, 'joint_valid_inters': joint_valid_inters,
                     'inv_trans': inv_trans, 'joint_valid': joint_valid}
        return inputs, targets, meta_info


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

    def evaluate(self, preds):

        print()
        print('Evaluation start...')

        # calculate PCK parameter
        # max_wh = 512
        # threshold = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        # threshold = [0.2]
        # threshold = np.array(threshold)

        gts = self.datalist
        pred_coord, inv_trans = preds['pred_coord'], preds['inv_trans']
        # print(len(pred_coord))
        # print(len(inv_trans))

        # assert len(gts) == len(joint_simdr_single_out)
        sample_num = len(pred_coord)

        mpjpe_rh = [[] for _ in range(self.joint_num * 2)]
        mpjpe_lh = [[] for _ in range(self.joint_num * 2)]
        mpjpe_th = [[] for _ in range(self.joint_num * 2)]
        mpjpe_ih = [[] for _ in range(self.joint_num * 2)]

        mpjpe_rh_2d = [[] for _ in range(self.joint_num * 2)]
        mpjpe_lh_2d = [[] for _ in range(self.joint_num * 2)]
        mpjpe_th_2d = [[] for _ in range(self.joint_num * 2)]
        mpjpe_ih_2d = [[] for _ in range(self.joint_num * 2)]

        for n in range(sample_num):
            data = gts[n]
            gt_hand_type, hand_type_valid = data['hand_type'], data['hand_type_valid']
            joint_valid = data['valid']

            # restore xy coordinates to original image space
            pred_joint_coord_img = pred_coord[n]

            # print()
            pred_joint_coord_img[:, 0] = pred_joint_coord_img[:, 0] / self.output_hm_shape[2] * self.input_img_shape[1]
            pred_joint_coord_img[:, 1] = pred_joint_coord_img[:, 1] / self.output_hm_shape[1] * self.input_img_shape[0]
            for j in range(self.joint_num * 2):
                pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans[n])

            # restore depth to original camera space
            pred_joint_coord_img[:, 2] = (pred_joint_coord_img[:, 2] / self.output_hm_shape[0] * 2 - 1) * (self.bbox_3d_size / 2)

            pred_joint_coord_img[self.joint_type['right'], 2] += data['abs_depth']['right']
            pred_joint_coord_img[self.joint_type['left'], 2] += data['abs_depth']['left']

            # back project to camera coordinate system
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img)
            gt_joint_coord = pixel2cam(data['img_coord'])
            # root joint alignment
            for h in ('right', 'left'):
                pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h], None, :]
                gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h], None, :]

            # mpjpe
            for j in range(self.joint_num * 2):
                if joint_valid[j]:
                    # if gt_hand_type == 'right' or gt_hand_type == 'left':
                    #     mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                    # else:
                    #     mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                    if gt_hand_type == 'right' and data['is_two'] == False:
                        mpjpe_rh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                        mpjpe_rh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                    elif gt_hand_type == 'left' and data['is_two'] == False:
                        mpjpe_lh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                        mpjpe_lh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                    elif data['is_two'] == True:
                        mpjpe_th_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                        mpjpe_th[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                    else:
                        mpjpe_ih_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                        mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))

            # img_path = data['img_path']
            # img = load_img(img_path)
            # vis_img = img.copy().transpose(2, 0, 1)
            # hand_type, bbox = data['hand_type'], data['bbox']
            # print(bbox)
        #     json_data['hand_type_list'].append(hand_type)
        #     json_data['bboxs'].append([bbox.tolist()])
        #     img = cv2.imread(img_path)
        #     cv2.imwrite(osp.join('./vis_img', str(n)+'.jpg'), img)
        # with open(osp.join('./vis_img', 'hand_type.json'), 'w') as obj:
        #     json.dump(json_data, obj)
            # gt_joint_coord = gt_joint_img_crop[n]
            # gt_joint_coord[:, 0] = gt_joint_coord[:, 0] / self.output_hm_shape[2] * self.input_img_shape[1]
            # gt_joint_coord[:, 1] = gt_joint_coord[:, 1] / self.output_hm_shape[1] * self.input_img_shape[0]
            # for j in self.joint_type['right']:
            #     gt_joint_coord[j, :2] = trans_point2d(gt_joint_coord[j, :2], inv_transs[n, 0])
            # for j in self.joint_type['left']:
            #     gt_joint_coord[j, :2] = trans_point2d(gt_joint_coord[j, :2], inv_transs[n, 1])
            # # restore depth to original camera space
            # gt_joint_coord[:, 2] = (gt_joint_coord[:, 2] / self.output_hm_shape[0] * 2 - 1) * (
            #             self.bbox_3d_size / 2)
            #
            # gt_coord_img = joint['img_coord']
            # vis_keypoints(vis_img, pred_joint_coord_img, joint_valid, self.skeleton, str(n)+'.jpg', save_path='./vis_img')
            # break
            # vis_keypoints(vis_img, gt_coord_img, joint_valid, self.skeleton, str(n) + '.jpg',save_path='./vis_img_gt')

        # eval_summary = 'PCK for each joint: \n'
        # pck_the = np.zeros((threshold.shape[0], self.joint_num * 2,))
        # for j in range(self.joint_num * 2):
        #     tot_err_j = np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j])))
        #     # print(tot_err_j.shape)
        #     for index in range(threshold.shape[0]):
        #         pck_the[index, j] = np.mean(tot_err_j[:]/512 <= threshold[index])
        #         # eval_summary += (str(threshold[index]) + ': %.2f, ' %pck_the[index, j])
        # print(eval_summary)
        # pck = np.mean(pck_the, axis=1)
        # print(pck)
        #
        # plt.xlim(0,50)
        # plt.ylim(0,1)
        # plt.plot(threshold, pck, color='red', linestyle='-', label='botnet')
        # plt.title('Avg. Keypoint Error 3D')
        # plt.xlabel('Aligned 3D Error (mm)')
        # plt.ylabel('PCK')
        # plt.legend(loc='lower right')
        # plt.grid()
        # plt.show()

        # 3D MPJPE
        tot_err = []
        eval_summary = 'MPJPE for each joint: \n'
        for j in self.joint_type['right']:
            tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh[j]), np.stack(mpjpe_th[j]), np.stack(mpjpe_ih[j]))))
            # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh[j]), np.stack(mpjpe_ih[j]))))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        for j in self.joint_type['left']:
            tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh[j]), np.stack(mpjpe_th[j]), np.stack(mpjpe_ih[j]))))
            # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh[j]), np.stack(mpjpe_ih[j]))))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        print(eval_summary)
        print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in self.joint_type['right']:
            mpjpe_rh[j] = np.mean(np.stack(mpjpe_rh[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_rh[j])
        print(eval_summary)
        print('MPJPE for right hand sequences: %.2f' % (np.mean(mpjpe_rh[:21])))
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in self.joint_type['left']:
            mpjpe_lh[j] = np.mean(np.stack(mpjpe_lh[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_lh[j])
        print(eval_summary)
        print('MPJPE for left hand sequences: %.2f' % (np.mean(mpjpe_lh[21:42])))
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num * 2):
            mpjpe_th[j] = np.mean(np.stack(mpjpe_th[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_th[j])
        print(eval_summary)
        print('MPJPE for two hand sequences: %.2f' % (np.mean(mpjpe_th)))
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num * 2):
            mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
        print(eval_summary)
        print('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih)))

        # 2D MPJPE
        tot_err = []
        eval_summary = 'MPJPE2d for each joint: \n'
        for j in self.joint_type['right']:
            tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh_2d[j]), np.stack(mpjpe_th_2d[j]), np.stack(mpjpe_ih_2d[j]))))
            # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_rh_2d[j]), np.stack(mpjpe_ih_2d[j]))))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        for j in self.joint_type['left']:
            tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh_2d[j]), np.stack(mpjpe_th_2d[j]), np.stack(mpjpe_ih_2d[j]))))
            # tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_lh_2d[j]), np.stack(mpjpe_ih_2d[j]))))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        print(eval_summary)
        print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
        print()

        eval_summary = 'MPJPE2d for each joint: \n'
        for j in self.joint_type['right']:
            mpjpe_rh_2d[j] = np.mean(np.stack(mpjpe_rh_2d[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_rh_2d[j])
        print(eval_summary)
        print('MPJPE for right hand sequences: %.2f' % (np.mean(mpjpe_rh_2d[:21])))
        print()

        eval_summary = 'MPJPE2d for each joint: \n'
        for j in self.joint_type['left']:
            mpjpe_lh_2d[j] = np.mean(np.stack(mpjpe_lh_2d[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_lh_2d[j])
        print(eval_summary)
        print('MPJPE for left hand sequences: %.2f' % (np.mean(mpjpe_lh_2d[21:42])))
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num * 2):
            mpjpe_th_2d[j] = np.mean(np.stack(mpjpe_th_2d[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_th_2d[j])
        print(eval_summary)
        print('MPJPE for two hand sequences: %.2f' % (np.mean(mpjpe_th_2d)))
        print()

        eval_summary = 'MPJPE2d for each joint: \n'
        for j in range(self.joint_num * 2):
            mpjpe_ih_2d[j] = np.mean(np.stack(mpjpe_ih_2d[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih_2d[j])
        print(eval_summary)
        print('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih_2d)))

    # def evaluate(self, preds):
    #     # for creating vedio-model input json
    #
    #     print()
    #     print('Evaluation start...')
    #
    #     pred_coord, inv_trans = preds['pred_coord'], preds['inv_trans']
    #
    #     input_json_path = osp.join(self.annot_path, 'STB_' + self.mode + '_data_input.json')
    #     json_data = []
    #
    #     n = 0
    #     for vedio_list in self.datalist:
    #
    #         data_list = vedio_list['datalist']
    #         json_vedio_data = []
    #         for data in data_list:
    #             # restore xy coordinates to original image space
    #             pred_joint_coord_img = pred_coord[n]
    #
    #             pred_joint_coord_img[:, 0] = pred_joint_coord_img[:, 0] / self.output_hm_shape[2] * self.input_img_shape[1]
    #             pred_joint_coord_img[:, 1] = pred_joint_coord_img[:, 1] / self.output_hm_shape[1] * self.input_img_shape[0]
    #             for j in range(self.joint_num):
    #                 pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans[n])
    #             # restore depth to original camera space
    #             pred_joint_coord_img[:, 2] = (pred_joint_coord_img[:, 2] / self.output_hm_shape[0] * 2 - 1) * (self.bbox_3d_size / 2)
    #
    #             pred_joint_coord_img[:, 2] = pred_joint_coord_img[:, 2] + data['abs_depth']
    #             # pred_joint_coord_img[self.joint_type['left'], 2] += data['abs_depth']['left']
    #
    #             # for creating vedio-model input json
    #             json_vedio_item = pred_joint_coord_img.tolist()
    #             json_vedio_data.append(json_vedio_item)
    #             n += 1
    #
    #         json_data.append(json_vedio_data)
    #
    #     with open(input_json_path, 'w') as obj:
    #         json.dump(json_data, obj)