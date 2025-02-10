import numpy as np
import torch
import torch.utils.data
import cv2
import os
import os.path as osp
from torchvision import transforms
from common_inter.utils.preprocessing import load_img, load_skeleton, process_bbox, get_aug_config, augmentation, \
    transform_input_to_output_space, generate_patch_image, trans_point2d
from common_inter.utils.transforms import world2cam, cam2pixel, pixel2cam
from common.utils.vis import vis_keypoints, vis_3d_keypoints, vis_kp_bbox
from pycocotools.coco import COCO
import json


class Dataset_STB(torch.utils.data.Dataset):
    # def __init__(self, cfg, mode):
    #     self.mode = mode
    #     self.root_path = '../data/STB'
    #     self.rootnet_output_path = '../data/STB/rootnet_output/rootnet_stb_output.json'
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
    #
    #     self.datalist = []
    #     self.annot_path = osp.join(self.root_path, 'annotations', 'STB_' + self.mode + '.json')
    #     db = COCO(self.annot_path)
    #
    #     self.transform = transforms.Compose([
    #         # transforms.ToPILImage(),
    #         # transforms.Resize((self.input_img_shape[1], self.input_img_shape[0])),
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #
    #     n = 0
    #     for aid in db.anns.keys():
    #         # n += 1
    #         # if n > 50:
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
    #         focal, princpt = np.array(cam_param['focal'], dtype=np.float32), np.array(cam_param['princpt'],
    #                                                                                   dtype=np.float32)
    #
    #         joint_img = np.array(ann['joint_img'], dtype=np.float32)
    #         joint_cam = np.array(ann['joint_cam'], dtype=np.float32)
    #         joint_valid = np.array(ann['joint_valid'], dtype=np.float32)
    #
    #         # transform single hand data to double hand data structure
    #         hand_type = ann['hand_type']
    #         joint_img_dh = np.zeros((self.joint_num * 2, 2), dtype=np.float32)
    #         joint_cam_dh = np.zeros((self.joint_num * 2, 3), dtype=np.float32)
    #         joint_valid_dh = np.zeros((self.joint_num * 2), dtype=np.float32)
    #         joint_img_dh[self.joint_type[hand_type]] = joint_img
    #         joint_cam_dh[self.joint_type[hand_type]] = joint_cam
    #         joint_valid_dh[self.joint_type[hand_type]] = joint_valid
    #         joint_img = joint_img_dh;
    #         joint_cam = joint_cam_dh;
    #         joint_valid = joint_valid_dh;
    #
    #         bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
    #         bbox = process_bbox(bbox, (img_height, img_width))
    #         abs_depth = joint_cam[self.root_joint_idx[hand_type], 2]  # single hand abs depth
    #
    #         cam_param = {'focal': focal, 'princpt': princpt}
    #         joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
    #         data = {'img_path': img_path, 'bbox': bbox, 'cam_param': cam_param, 'joint': joint, 'hand_type': hand_type,
    #                 'abs_depth': abs_depth}
    #         self.datalist.append(data)
    #
    # def __len__(self):
    #     return len(self.datalist)
    #
    # def __getitem__(self, idx):
    #     data = self.datalist[idx]
    #     img_path, bbox, joint, hand_type = data['img_path'], data['bbox'], data['joint'], data['hand_type']
    #     joint_cam = joint['cam_coord'].copy();
    #     joint_img = joint['img_coord'].copy();
    #     joint_valid = joint['valid'].copy();
    #     # hand_type = self.handtype_str2array(hand_type)
    #     joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)
    #
    #     # image load
    #     img = load_img(img_path)
    #     # augmentation
    #     img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid,
    #                                                                        hand_type, self.mode, self.joint_type)
    #     img = self.transform(img.astype(np.float32)) / 255.
    #     rel_root_depth = np.zeros((1), dtype=np.float32)
    #     root_valid = np.zeros((1), dtype=np.float32)
    #     # transform to output heatmap space
    #     joint_coord, joint_valid, rel_root_depth = transform_input_to_output_space(joint_coord, joint_valid,
    #                                                                                            rel_root_depth,
    #                                                                                            self.root_joint_idx,
    #                                                                                            self.joint_type)
    #
    #     inputs = {'img': img}
    #     targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
    #     meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'inv_trans': inv_trans, 'hand_type_valid': 1}
    #     return inputs, targets, meta_info
    #
    #
    # def evaluate(self, preds):
    #
    #     gts = self.datalist
    #     preds_joint_coord, preds_rel_root_depth, preds_hand_type, inv_trans = preds['joint_coord'], preds[
    #         'rel_root_depth'], preds['hand_type'], preds['inv_trans']
    #     assert len(gts) == len(preds_joint_coord)
    #     sample_num = len(gts)
    #
    #     mpjpe = [[] for _ in range(self.joint_num)]  # treat right and left hand identical
    #     mpjpe_2d = [[] for _ in range(self.joint_num)]
    #     for n in range(sample_num):
    #         data = gts[n]
    #         bbox, cam_param, joint, gt_hand_type = data['bbox'], data['cam_param'], data['joint'], data['hand_type']
    #         focal = cam_param['focal']
    #         princpt = cam_param['princpt']
    #         gt_joint_coord = joint['cam_coord']
    #         joint_valid = joint['valid']
    #
    #         # restore coordinates to original space
    #         pred_joint_coord_img = preds_joint_coord[n].copy()
    #         pred_joint_coord_img[:, 0] = pred_joint_coord_img[:, 0] / self.output_hm_shape[2] * self.input_img_shape[1]
    #         pred_joint_coord_img[:, 1] = pred_joint_coord_img[:, 1] / self.output_hm_shape[1] * self.input_img_shape[0]
    #         for j in range(self.joint_num * 2):
    #             pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans[n])
    #         pred_joint_coord_img[:, 2] = (pred_joint_coord_img[:, 2] / self.output_hm_shape[0] * 2 - 1) * (
    #                     self.bbox_3d_size / 2)
    #         pred_joint_coord_img[:, 2] = pred_joint_coord_img[:, 2] + data['abs_depth']
    #
    #         # back project to camera coordinate system
    #         pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)
    #
    #         # root joint alignment
    #         pred_joint_coord_cam[self.joint_type['right']] = pred_joint_coord_cam[self.joint_type['right']] - pred_joint_coord_cam[self.root_joint_idx['right'], None, :]
    #         pred_joint_coord_cam[self.joint_type['left']] = pred_joint_coord_cam[self.joint_type['left']] - pred_joint_coord_cam[
    #                                                                                        self.root_joint_idx['left'],
    #                                                                                        None, :]
    #         gt_joint_coord[self.joint_type['right']] = gt_joint_coord[self.joint_type['right']] - gt_joint_coord[
    #                                                                                               self.root_joint_idx[
    #                                                                                                   'right'], None, :]
    #         gt_joint_coord[self.joint_type['left']] = gt_joint_coord[self.joint_type['left']] - gt_joint_coord[
    #                                                                                             self.root_joint_idx[
    #                                                                                                 'left'], None, :]
    #
    #         # select right or left hand using groundtruth hand type
    #         pred_joint_coord_cam = pred_joint_coord_cam[self.joint_type[gt_hand_type]]
    #         gt_joint_coord = gt_joint_coord[self.joint_type[gt_hand_type]]
    #         joint_valid = joint_valid[self.joint_type[gt_hand_type]]
    #
    #         # mpjpe save
    #         for j in range(self.joint_num):
    #             if joint_valid[j]:
    #                 mpjpe[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))
    #                 mpjpe_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j, :2] - gt_joint_coord[j, :2]) ** 2)))
    #
    #         # img_path = data['img_path']
    #         # img = load_img(img_path)
    #         # vis_img = img.copy().transpose(2, 0, 1)
    #         # #
    #         # # gt_coord_img = joint['img_coord']
    #         # vis_keypoints(vis_img, pred_joint_coord_img, joint['valid'], self.skeleton, str(n) + '.jpg',
    #         #               save_path='./vis_img_STB')
    #
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
    #         joint_name = self.skeleton[j]['name']
    #         eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
    #         tot_err.append(tot_err_j)
    #     print(eval_summary)
    #     print('MPJPE2d for all hand sequences: %.2f' % (np.mean(tot_err)))
    #     print()

    def __init__(self, cfg, mode):
        # for creating vedio-model input json
        self.mode = 'test'
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

        self.transform = transforms.Compose([
            transforms.ToTensor(),
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
        img = self.transform(img.astype(np.float32)) /255
        rel_root_depth = np.zeros((1),dtype=np.float32)

        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth = transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, self.root_joint_idx, self.joint_type)




        inputs = {'img': img}
        targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}#, 'js_single': js_single}
        meta_info = {'joint_valid': joint_valid, 'inv_trans': inv_trans, 'hand_type_valid': 1}#, 'js_valid_singles': js_valid_single}
        return inputs, targets, meta_info


    def evaluate(self, preds):
        # for creating vedio-model input json

        print()
        print('Evaluation start...')

        pred_coord, inv_trans = preds['joint_coord'], preds['inv_trans']

        input_json_path = osp.join(self.annot_path, 'STB_' + self.mode + '_data_input_inter.json')
        json_data = []

        n = 0
        mpjpe = [[] for _ in range(self.joint_num)]
        for vedio_list in self.datalist:

            data_list = vedio_list['datalist']
            json_vedio_data = []
            for data in data_list:
                bbox, cam_param, joint, gt_hand_type = data['bbox'], data['cam_param'], data['joint'], data['hand_type']
                focal = cam_param['focal']
                princpt = cam_param['princpt']
                gt_joint_coord = joint['cam_coord']
                joint_valid = joint['valid']
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
                pred_joint_coord_img = pred_joint_coord_img[self.joint_type[gt_hand_type]]

                # for creating vedio-model input json
                json_vedio_item = pred_joint_coord_img.tolist()
                json_vedio_data.append(json_vedio_item)
                n += 1

                # back project to camera coordinate system
                # pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)
                #
                # # root joint alignment
                # pred_joint_coord_cam[self.joint_type['right']] = pred_joint_coord_cam[self.joint_type['right']] - pred_joint_coord_cam[self.root_joint_idx['right'], None, :]
                # pred_joint_coord_cam[self.joint_type['left']] = pred_joint_coord_cam[self.joint_type['left']] - pred_joint_coord_cam[self.root_joint_idx['left'],None, :]
                # gt_joint_coord[self.joint_type['right']] = gt_joint_coord[self.joint_type['right']] - gt_joint_coord[self.root_joint_idx['right'],None, :]
                # gt_joint_coord[self.joint_type['left']] = gt_joint_coord[self.joint_type['left']] - gt_joint_coord[self.root_joint_idx['left'], None,:]
                #
                # pred_joint_coord_cam = pred_joint_coord_cam[self.joint_type[gt_hand_type]]
                # gt_joint_coord = gt_joint_coord[self.joint_type[gt_hand_type]]
                # joint_valid = joint_valid[self.joint_type[gt_hand_type]]
                #
                # # mpjpe save
                # for j in range(self.joint_num):
                #     if joint_valid[j]:
                #         mpjpe[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))

            json_data.append(json_vedio_data)

        with open(input_json_path, 'w') as obj:
            json.dump(json_data, obj)

        # eval_summary = 'MPJPE for each joint: \n'
        # for j in range(self.joint_num):
        #     mpjpe[j] = np.mean(np.stack(mpjpe[j]))
        #     joint_name = self.skeleton[j]['name']
        #     eval_summary += (joint_name + ': %.2f, ' % mpjpe[j])
        # print(eval_summary)
        # print('MPJPE: %.2f' % (np.mean(mpjpe)))