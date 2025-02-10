import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp

from pycocotools.coco import COCO
import sys
sys.path.append('..')
from common.utils.coco_diy import COCO as COCO_diy

from common.utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from common.utils.transforms import world2cam, cam2pixel, pixel2cam
import json
from torchvision import transforms



def load_data(data_json_path, cameras_json_path, joints_json_path, img_dir, skeleton):
    joint_num = 21  # single hand
    root_joint_idx = {'right': 20, 'left': 41}
    joint_type = {'right': np.arange(0, joint_num), 'left': np.arange(joint_num, joint_num * 2)}
    datalist = []
    datalist_rh = []
    datalist_lh = []
    datalist_ih = []
    datalist_th = []
    # load annotation
    print("Load annotation from  " + data_json_path)
    db = COCO(data_json_path)
    with open(cameras_json_path) as f:
        cameras = json.load(f)
    with open(joints_json_path) as f:
        joints = json.load(f)

    rh = 0
    lh = 0
    th = 0
    ih = 0

    for aid in db.anns.keys():
        ann = db.anns[aid]
        image_id = ann['image_id']
        img = db.loadImgs(image_id)[0]

        capture_id = img['capture']
        seq_name = img['seq_name']
        cam = img['camera']
        frame_idx = img['frame_idx']

        img_path = osp.join(img_dir, img['file_name'])

        campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
            cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
        focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
            cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
        joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
        joint_cam = world2cam(joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
        joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

        joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(joint_num * 2)
        # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
        joint_valid[joint_type['right']] *= joint_valid[root_joint_idx['right']]
        joint_valid[joint_type['left']] *= joint_valid[root_joint_idx['left']]

        # bone_valid = self.calBonesValid(joint_valid)

        hand_type = ann['hand_type']
        hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

        img_width, img_height = img['width'], img['height']
        bboxs = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h

        abs_depth = {'right': joint_cam[root_joint_idx['right'], 2],
                         'left': joint_cam[root_joint_idx['left'], 2]}

        # print(bbox)
        cam_param = {'focal': focal, 'princpt': princpt}
        joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
        data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bboxs, 'joint': joint,
                'hand_type': hand_type, 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth,
                'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
                'is_two': False}

        if hand_type == 'right':
            # rh += 1
            # if (rh > 50):
            #     continue
            bboxs[0] = process_bbox(bboxs[0], (img_height, img_width))  # houxuchuli
            data['bbox'] = bboxs[0]
            datalist_rh.append(data)
        elif hand_type == 'left':
            # lh += 1
            # if (lh > 50):
            #     continue
            bboxs[0] = process_bbox(bboxs[0], (img_height, img_width))
            data['bbox'] = bboxs[0]
            datalist_lh.append(data)
        elif hand_type == 'two':
            # th += 1
            # if (th > 50):
            #     continue
            joint_valid_right = np.zeros((joint_num * 2), dtype=np.float32)
            joint_valid_right[joint_type['right']] = joint_valid[joint_type['right']]
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid_right}
            bboxs[0] = process_bbox(bboxs[0], (img_height, img_width))
            data_right = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bboxs[0],
                          'joint': joint,
                          'hand_type': 'right', 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth,
                          'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
                          'is_two': True}

            joint_valid_left = np.zeros((joint_num * 2), dtype=np.float32)
            joint_valid_left[joint_type['left']] = joint_valid[joint_type['left']]
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid_left}
            bboxs[1] = process_bbox(bboxs[1], (img_height, img_width))
            data_left = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bboxs[1],
                         'joint': joint,
                         'hand_type': 'left', 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth,
                         'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
                         'is_two': True}

            datalist_th.append(data_right)
            datalist_th.append(data_left)
        else:
            # ih += 1
            # if (ih > 50):
            #     continue
            bboxs[0] = process_bbox(bboxs[0], (img_height, img_width))
            data['bbox'] = bboxs[0]
            datalist_ih.append(data)



    datalist = datalist_rh + datalist_lh + datalist_th + datalist_ih

    print('Number of annotations in right hand sequences: ' + str(len(datalist_rh)))
    print('Number of annotations in left hand sequences: ' + str(len(datalist_lh)))
    print('Number of annotations in two hand sequences: ' + str(len(datalist_th)))
    print('Number of annotations in interacting hand sequences: ' + str(len(datalist_ih)))
    return datalist