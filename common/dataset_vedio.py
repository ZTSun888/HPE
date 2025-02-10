import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp

# from pycocotools.coco import COCO
import sys
sys.path.append('..')
from common.utils.coco_diy import COCO
from common.utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from common.utils.transforms import world2cam, cam2pixel, pixel2cam
import json
from torchvision import transforms
from common.utils.vis import vis_keypoints
from common.utils.bone import getbonejs
from main.config import cfg
import random

#
# class cfg:
#     output_hm_shape = (64, 64, 64)


class DatasetVedio(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, annot_subset, pad, inference=False):
        # for creating vedio-model input json
        self.mode = mode   # train, test, val
        self.pad = pad
        self.inference = inference
        if self.inference:
            self.opt_coord_list = []

        self.annot_subset = annot_subset  # all, human_annot, machine_annot
        self.img_path = '../data/InterHand2.6M_vedio/images'
        self.annot_path = '../data/InterHand2.6M_vedio/annotations'
        self.joint_num = 21  # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num * 2)}
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num * 2)


        # load annotation
        print("Load annotation from  " + osp.join(self.annot_path, self.annot_subset))
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data_bbox.json')) as f:
            dbs = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)
        # with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data_input.json')) as f:
        #     pred_dbs = json.load(f)

        # for InterHand model predictions as input
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data_input_inter.json')) as f:
            pred_dbs = json.load(f)

        print("Get bbox from groundtruth annotation")

        i = 0
        self.idx_list = []
        vedio_idx = 0
        self.datalist = []
        for vedio_dict in dbs:
            # if vedio_idx > 1:
            #     break
            img_idx = 0
            vedio_path = vedio_dict['vedio_path']
            datalist = []
            db = COCO(vedio_dict['data'])
            if self.inference:
                self.opt_coord_list.append([])

            for aid in db.anns.keys():
                # if i >= 1:
                #     break
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]

                capture_id = img['capture']
                # if capture_id != 0:
                #     break
                seq_name = img['seq_name']
                cam = img['camera']
                frame_idx = img['frame_idx']

                img_path = osp.join(self.img_path, self.mode, img['file_name'])

                campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
                    cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
                focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
                    cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
                joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
                joint_cam = world2cam(joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
                joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

                img_width, img_height = img['width'], img['height']

                joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(self.joint_num * 2)
                # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
                joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
                joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]

                hand_type = ann['hand_type']
                hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

                pred_joint_coord_img = np.array(pred_dbs[vedio_idx][img_idx], dtype=np.float32)

                input_root_coord = {'right': pred_joint_coord_img[self.root_joint_idx['right']], 'left': pred_joint_coord_img[self.root_joint_idx['left']]}

                cam_param = {'focal': focal, 'princpt': princpt}
                joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid, 'pred_joint_coord_img': pred_joint_coord_img}
                data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param,
                        'joint': joint,
                        'hand_type': hand_type, 'hand_type_valid': hand_type_valid, 'input_root_coord': input_root_coord,
                        'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
                        'img_width': img_width, 'img_height': img_height}
                datalist.append(data)
                self.idx_list.append([vedio_idx, img_idx])
                img_idx += 1

                # i += 1

            vedio_list = {'vedio_path': vedio_path, 'datalist': datalist}
            self.datalist.append(vedio_list)
            vedio_idx += 1
            # break

        print('total img num: ' + str(len(self.idx_list)))


    def __len__(self):
        # for creating vedio-model input json
        return len(self.idx_list)

    def __getitem__(self, idx):
        # for creating vedio-model input json
        vedio_idx, img_idx = self.idx_list[idx]
        data = self.datalist[vedio_idx]['datalist'][img_idx]
        img_width, img_height = data['img_width'], data['img_height']

        input_list = []
        joint_valid_list = []

        gt_joint_coord_list = []

        if not self.inference:
            for i in range(img_idx - self.pad + 1, img_idx + 1):
                if i < 0:
                    data = self.datalist[vedio_idx]['datalist'][0]
                else:
                    data = self.datalist[vedio_idx]['datalist'][i]
                joint = data['joint']
                input_list.append(joint['pred_joint_coord_img'].copy())
                joint_valid_list.append(joint['valid'].copy())

                joint_cam = joint['cam_coord'].copy();
                joint_img = joint['img_coord'].copy();
                joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)
                gt_joint_coord_list.append(joint_coord.copy())

            input_list = np.array(input_list, dtype=np.float32)
            gt_joint_coord_list = np.array(gt_joint_coord_list, dtype=np.float32)
            for i in range(input_list.shape[0]):
                for h in ('right', 'left'):
                    input_list[i, self.joint_type[h], 2] = input_list[i, self.joint_type[h], 2] - input_list[i, self.root_joint_idx[h], 2]
                    gt_joint_coord_list[i, self.joint_type[h], 2] = gt_joint_coord_list[i, self.joint_type[h], 2] - gt_joint_coord_list[i, self.root_joint_idx[h], 2]

        # if not self.inference:
        #     for i in range(img_idx - self.pad + 1, img_idx):
        #         if i < 0:
        #             data = self.datalist[vedio_idx]['datalist'][0]
        #         else:
        #             data = self.datalist[vedio_idx]['datalist'][i]
        #         joint = data['joint']
        #         joint_cam = joint['cam_coord'].copy();
        #         joint_img = joint['img_coord'].copy();
        #         # joint_valid = joint['valid'].copy();
        #         joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)
        #         input_list.append(joint_coord.copy())
        #         joint_valid_list.append(joint['valid'].copy())
        #
        #     data = self.datalist[vedio_idx]['datalist'][img_idx]
        #     joint = data['joint']
        #     input_list.append(joint['pred_joint_coord_img'].copy())
        #     joint_valid_list.append(joint['valid'].copy())
        #     input_list = np.array(input_list, dtype=np.float32)
        #     for i in range(input_list.shape[0]):
        #         for h in ('right', 'left'):
        #             input_list[i, self.joint_type[h], 2] = input_list[i, self.joint_type[h], 2] - input_list[i, self.root_joint_idx[h], 2]

        else:
            for i in range(img_idx - self.pad + 1, img_idx):
                if i < 0:
                    data = self.datalist[vedio_idx]['datalist'][0]
                    joint = data['joint']
                    input_list.append(joint['pred_joint_coord_img'].copy())
                else:
                    opt_coord = self.opt_coord_list[vedio_idx][i]
                    input_list.append(opt_coord.copy())
                    data = self.datalist[vedio_idx]['datalist'][i]
                    joint = data['joint']
                joint_valid_list.append(joint['valid'].copy())
            data = self.datalist[vedio_idx]['datalist'][img_idx]
            joint = data['joint']
            input_list.append(joint['pred_joint_coord_img'].copy())
            input_list = np.array(input_list, dtype=np.float32)
            for i in range(input_list.shape[0]):
                for h in ('right', 'left'):
                    input_list[i, self.joint_type[h], 2] = input_list[i, self.joint_type[h], 2] - input_list[i, self.root_joint_idx[h], 2]

        joint_valid_list = np.array(joint_valid_list, dtype=np.float32)
        # joint_valid = np.ones(self.joint_num * 2)
        # for i in range(joint_valid_list.shape[0]):
        #     joint_valid[:] *= joint_valid_list[i, :]
        joint_valid = joint_valid_list[-1, :]


        # print(bbox)
        # input_list[:, :, 2] /= 1000
        # input_list = (input_list - (512 / 2)) / (512/2)

        # norm to [-1, 1]
        input_list[:, :, 0] = (input_list[:, :, 0] - (img_width / 2)) / (img_width / 2)
        input_list[:, :, 1] = (input_list[:, :, 1] - (img_height / 2)) / (img_height / 2)
        input_list[:, :, 2] = input_list[:, :, 2] / 200

        inputs = input_list
        # targets = gt_joint_coord
        # gt_joint_coord_list[:, :, 2] /= 1000
        # gt_joint_coord_list = (gt_joint_coord_list - (512 / 2)) / (512/2)
        # targets = gt_joint_coord_list
        meta_info = {'joint_valid': joint_valid, 'vedio_idx': vedio_idx, 'bone_valid': self.get_bone_valid(joint_valid)}

        return inputs,  meta_info

    def append_opt_coord_list(self, vedio_idx, out):
        for i in range(out.shape[0]):
            self.opt_coord_list[vedio_idx[i]].append(out[i])

    def get_bone_valid(self, joint_valid):
        bone_indexs = cfg.bone_index
        bone_valid = np.zeros(cfg.bone_num * 2)
        for i in range(len(bone_indexs)):
            bone_valid[i] = joint_valid[bone_indexs[i][0]] * joint_valid[bone_indexs[i][1]]
        return bone_valid

    def cal_motion_loss(self, trail, pad):
        time_interval = 8
        motion_loss_opt = np.zeros((len(trail) - time_interval), dtype=np.float)

        for i in range(0, len(trail) - time_interval):
            # motion_loss_opt[i] = 0

            motion_loss_opt[i] = np.sum(abs(np.cross(trail[i, :2], trail[i + time_interval, :2])))
            # abs(np.cross(trail[i + t, :2], trail[i + t + time_interval, :2])))

        # motion_loss_opt /= pad - time_interval

        ml_opt = np.mean(motion_loss_opt)
        return ml_opt

    def cal_acc(self, trail_opt):

        accel_opt = trail_opt[:-2] - 2 * trail_opt[1:-1] + trail_opt[2:]
        normed_opt = np.linalg.norm(accel_opt, axis=2)
        new_vis_opt = np.ones(len(normed_opt), dtype=bool)
        acc_opt = np.mean(np.mean(normed_opt[new_vis_opt], axis=1))

        return acc_opt



    def evaluate(self, preds):

        print()
        print('Evaluation start...')

        # gts = self.datalist
        out = preds['pred_coord']
        # print(len(out))

        # assert len(gts) == len(joint_simdr_single_out)
        # sample_num = len(gts)

        mpjpe_rh = [[] for _ in range(self.joint_num * 2)]
        mpjpe_lh = [[] for _ in range(self.joint_num * 2)]
        mpjpe_th = [[] for _ in range(self.joint_num * 2)]
        mpjpe_ih = [[] for _ in range(self.joint_num * 2)]

        mpjpe_rh_2d = [[] for _ in range(self.joint_num * 2)]
        mpjpe_lh_2d = [[] for _ in range(self.joint_num * 2)]
        mpjpe_th_2d = [[] for _ in range(self.joint_num * 2)]
        mpjpe_ih_2d = [[] for _ in range(self.joint_num * 2)]
        # # mrrpe = []
        # acc_hand_cls = 0;
        # hand_cls_cnt = 0;
        n = 0

        video_idx = 0
        motion_loss_opt_list = []
        motion_loss_pred_list = []
        acc_list = []
        acc_gt_list = []


        for vedio_list in self.datalist:
            data_list = vedio_list['datalist']

            video_len = len(data_list)
            video_hand_type = data_list[0]['hand_type']
            if video_hand_type == 'right' or video_hand_type == 'left':
                opt_trail = np.zeros((video_len, 21, 3), dtype=np.float32)
                img_trail = np.zeros((video_len, 21, 3), dtype=np.float32)
            else:
                opt_trail = np.zeros((video_len, 42, 3), dtype=np.float32)
                img_trail = np.zeros((video_len, 42, 3), dtype=np.float32)
            coord_idx = 0

            for data in data_list:
                cam_param, joint, gt_hand_type, hand_type_valid = data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
                focal = cam_param['focal']
                princpt = cam_param['princpt']
                gt_joint_coord = joint['cam_coord']
                joint_valid = joint['valid']

                pred_joint_coord_img = out[n]

                pred_joint_coord_img_input = data['input_root_coord']
                input_joint_coord = joint['pred_joint_coord_img']

                img_width, img_height = data['img_width'], data['img_height']
                pred_joint_coord_img[:, 0] = (pred_joint_coord_img[:, 0] * (img_width / 2)) + (img_width / 2)
                pred_joint_coord_img[:, 1] = (pred_joint_coord_img[:, 1] * (img_height / 2)) + (img_height / 2)
                pred_joint_coord_img[:, 2] = pred_joint_coord_img[:, 2] * 200
                # print(pred_joint_coord_img)
                for h in ('right', 'left'):
                    pred_joint_coord_img[self.joint_type[h], 2] = pred_joint_coord_img[self.joint_type[h], 2] + pred_joint_coord_img_input[h][2]


                # # back project to camera coordinate system
                #
                pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt).copy()
                input_joint_coord_cam = pixel2cam(input_joint_coord, focal, princpt).copy()

                # # root joint alignment
                # for h in ('right', 'left'):
                #     pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h]]
                #     gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h]]

                # mpjpe
                for j in range(self.joint_num * 2):
                    if joint_valid[j]:
                        if gt_hand_type == 'right':
                            mpjpe_rh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                            mpjpe_rh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                        elif gt_hand_type == 'left':
                            mpjpe_lh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                            mpjpe_lh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                        elif gt_hand_type == 'two':
                            mpjpe_th_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                            mpjpe_th[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
                        else:
                            mpjpe_ih_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
                            mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))

                if video_hand_type == 'right' or video_hand_type == 'left':
                    opt_trail[coord_idx] = pred_joint_coord_cam[self.joint_type[video_hand_type]]
                    img_trail[coord_idx] = input_joint_coord_cam[self.joint_type[video_hand_type]]
                else:
                    opt_trail[coord_idx] = pred_joint_coord_cam
                    img_trail[coord_idx] = input_joint_coord_cam

                coord_idx += 1

                n += 1

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
                # #
                # # gt_coord_img = joint['img_coord']
                # # pred_joint_coord_img = cam2pixel(pred_joint_coord_cam, focal, princpt)
                # # for h in ('right', 'left'):
                # #     pred_joint_coord_img[self.joint_type[h],:2] = pred_joint_coord_img[self.joint_type[h], :2] + gt_coord_img[self.root_joint_idx[h], :]
                # # print(pred_joint_coord_img)
                # vis_keypoints(vis_img, pred_joint_coord_img, joint_valid, self.skeleton, str(n)+'.jpg', save_path='./vis_img_vedio')
                # gt_joint_coord = joint['cam_coord']
                # gt_coord_img = cam2pixel(gt_joint_coord, focal, princpt)[:, :2]
                # vis_keypoints(vis_img, gt_coord_img, joint_valid, self.skeleton, str(n) + '.jpg',save_path='./vis_img_gt')
            motion_loss_opt_list.append(self.cal_motion_loss(opt_trail, self.pad))
            motion_loss_pred_list.append(self.cal_motion_loss(img_trail, self.pad))
            acc_list.append(self.cal_acc(opt_trail))
            acc_gt_list.append(self.cal_acc(img_trail))
            # print(acc_list[0])
            video_idx += 1

        # motion_loss_opt = np.mean(np.array(motion_loss_opt_list))
        # motion_loss_pred = np.mean(np.array(motion_loss_pred_list))
        # print('opt motion loss:' + str(motion_loss_opt))
        # print('pred motion loss:' + str(motion_loss_pred))
        acc = np.mean(np.array(acc_list))
        acc_gt = np.mean(np.array(acc_gt_list))
        print('acc:' + str(acc))
        print('acc_gt:' + str(acc_gt))




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


