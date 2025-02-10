import os
import os.path as osp
import cv2
import numpy as np
import matplotlib

matplotlib.use('tkagg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from main.config import cfg
from PIL import Image, ImageDraw
from einops import rearrange


def get_keypoint_rgb(skeleton):
    rgb_dict = {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb4'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index4'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle4'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring4'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky4'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
        # if joint_id < 21:
        #     rgb_dict[joint_name] = (0, 0, 255)
        # else:
        #     rgb_dict[joint_name] = (255, 0, 0)
    return rgb_dict


def visTwoBbox(img, bbox, filename, save_path):
    line_width = 3
    _img = Image.fromarray(img.astype('uint8'))
    draw = ImageDraw.Draw(_img)
    # line_rgb = (0, 255, 0)
    # draw.line([(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1])], fill=line_rgb, width=line_width)
    # draw.line([(bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])], fill=line_rgb, width=line_width)
    # draw.line([(bbox[0], bbox[1] + bbox[3]), (bbox[0] + bbox[2], bbox[1] + bbox[3])], fill=line_rgb, width=line_width)
    # draw.line([(bbox[0], bbox[1]), (bbox[0], bbox[1] + bbox[3])], fill=line_rgb, width=line_width)
    line_rgb = (0, 255, 0)
    right_bbox = bbox[0]
    draw.line([(right_bbox[0], right_bbox[1]), (right_bbox[0] + right_bbox[2], right_bbox[1])], fill=line_rgb,
              width=line_width)
    draw.line([(right_bbox[0] + right_bbox[2], right_bbox[1]),
               (right_bbox[0] + right_bbox[2], right_bbox[1] + right_bbox[3])], fill=line_rgb, width=line_width)
    draw.line([(right_bbox[0], right_bbox[1] + right_bbox[3]),
               (right_bbox[0] + right_bbox[2], right_bbox[1] + right_bbox[3])], fill=line_rgb, width=line_width)
    draw.line([(right_bbox[0], right_bbox[1]), (right_bbox[0], right_bbox[1] + right_bbox[3])], fill=line_rgb,
              width=line_width)

    line_rgb = (255, 0, 0)
    left_bbox = bbox[1]
    draw.line([(left_bbox[0], left_bbox[1]), (left_bbox[0] + left_bbox[2], left_bbox[1])], fill=line_rgb,
              width=line_width)
    draw.line([(left_bbox[0] + left_bbox[2], left_bbox[1]),
               (left_bbox[0] + left_bbox[2], left_bbox[1] + left_bbox[3])], fill=line_rgb, width=line_width)
    draw.line([(left_bbox[0], left_bbox[1] + left_bbox[3]),
               (left_bbox[0] + left_bbox[2], left_bbox[1] + left_bbox[3])], fill=line_rgb, width=line_width)
    draw.line([(left_bbox[0], left_bbox[1]), (left_bbox[0], left_bbox[1] + left_bbox[3])], fill=line_rgb,
              width=line_width)

    # draw.line([(0,0), (img.shape[1], 0)], fill=(255,255,255), width= 5)
    # draw.line([(0,0), (0, img.shape[0])], fill=(255,255,255), width= 5)

    _img.save(osp.join(save_path, filename))


def visBbox(img, bbox, filename, save_path):
    line_width = 3
    _img = Image.fromarray(img.astype('uint8'))
    draw = ImageDraw.Draw(_img)
    line_rgb = (0, 0, 0)
    draw.line([(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1])], fill=line_rgb,
              width=line_width)
    draw.line([(bbox[0] + bbox[2], bbox[1]),
               (bbox[0] + bbox[2], bbox[1] + bbox[3])], fill=line_rgb, width=line_width)
    draw.line([(bbox[0], bbox[1] + bbox[3]),
               (bbox[0] + bbox[2], bbox[1] + bbox[3])], fill=line_rgb, width=line_width)
    draw.line([(bbox[0], bbox[1]), (bbox[0], bbox[1] + bbox[3])], fill=line_rgb,
              width=line_width)
    _img.save(osp.join(save_path, filename))


def vis_keypoints(img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3, save_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.transpose(1, 2, 0).astype('uint8'))
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name],
                      width=line_width)
        if score[i] > score_thr:
            draw.ellipse(
                (kps[i][0] - circle_rad, kps[i][1] - circle_rad, kps[i][0] + circle_rad, kps[i][1] + circle_rad),
                fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0] - circle_rad, kps[pid][1] - circle_rad, kps[pid][0] + circle_rad,
                          kps[pid][1] + circle_rad), fill=rgb_dict[parent_joint_name])

    if save_path is None:
        _img.save(osp.join(cfg.vis_dir, filename))
    else:
        _img.save(osp.join(save_path, filename))


def vis_kp_bbox(img, kps, score, skeleton, bboxs, filename, score_thr=0.4, line_width=3, circle_rad=3, save_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.transpose(1, 2, 0).astype('uint8'))
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name],
                      width=line_width)
        if score[i] > score_thr:
            draw.ellipse(
                (kps[i][0] - circle_rad, kps[i][1] - circle_rad, kps[i][0] + circle_rad, kps[i][1] + circle_rad),
                fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0] - circle_rad, kps[pid][1] - circle_rad, kps[pid][0] + circle_rad,
                          kps[pid][1] + circle_rad), fill=rgb_dict[parent_joint_name])


    line_rgb =  (0, 255, 0)
    for bbox in bboxs:
        draw.line([(bbox[0], bbox[1]),(bbox[0]+bbox[2], bbox[1])], fill=line_rgb, width=line_width)
        draw.line([(bbox[0]+bbox[2], bbox[1]), (bbox[0]+ bbox[2], bbox[1]+bbox[3])], fill=line_rgb, width=line_width)
        draw.line([(bbox[0], bbox[1]+bbox[3]), (bbox[0]+bbox[2], bbox[1] + bbox[3])], fill=line_rgb, width=line_width)
        draw.line([(bbox[0], bbox[1]), (bbox[0], bbox[1] + bbox[3])], fill=line_rgb, width=line_width)

    if save_path is None:
        _img.save(osp.join(cfg.vis_dir, filename))
    else:
        _img.save(osp.join(save_path, filename))


def vis_3d_keypoints(kps_3d, score, skeleton, filename, score_thr=0.5, line_width=3, circle_rad=3, save_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    # plt.xlim(-160,0)
    # plt.ylim(50,300)
    # ax.axis([50,300, -160,0])
    # ax.plot.set_ylim([50,300])
    # ax.set_xlim3d(xmin=50, xmax=300)
    # ax.set_ylim3d(ymin=-160, ymax=0)
    # ax.set_zlim3d(zmin=-450, zmax=-200)
    rgb_dict = get_keypoint_rgb(skeleton)

    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i, 0], kps_3d[pid, 0]])
        y = np.array([kps_3d[i, 1], kps_3d[pid, 1]])
        z = np.array([kps_3d[i, 2], kps_3d[pid, 2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c=np.array(rgb_dict[parent_joint_name]) / 255., linewidth=line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i, 0], kps_3d[i, 2], -kps_3d[i, 1], c=np.array(rgb_dict[joint_name]).reshape(1, 3) / 255.,
                       marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid, 0], kps_3d[pid, 2], -kps_3d[pid, 1],
                       c=np.array(rgb_dict[parent_joint_name]).reshape(1, 3) / 255., marker='o')

    # plt.show()
    # cv2.waitKey(0)

    # fig.savefig(osp.join(cfg.vis_dir, filename), dpi=fig.dpi)
    if save_path is None:
        fig.savefig(osp.join(cfg.vis_dir, filename), dpi=fig.dpi)
    else:
        fig.savefig(osp.join(save_path, filename), dpi=fig.dpi)
    plt.close()


def softmax(arr):
    arr1 = arr-arr.max()
    arr1_exp = np.exp(arr1)
    arr1_exp_sum = arr1_exp.sum()

    arr_softmax = arr1_exp / arr1_exp_sum
    return arr_softmax


# def vis_simdr(pred_simdr, gt_simdr, file_idx, save_dir):
def vis_simdr(pred_simdr, file_idx, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    x = [i for i in range(448)]
    file_name = file_idx.split('.')[0]
    for i in range(42):
        fig = plt.figure()
        for j in range(3):
            plt.subplot(3, 1, j+1)
            plt.plot(x, softmax(pred_simdr[i, j]),color='r', linestyle='-')
            # plt.plot(x, gt_simdr[i, j], color='g', linestyle='-.')
        save_path = os.path.join(save_dir, file_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        fig.savefig(os.path.join(save_path, str(i)+'.jpg'))
        plt.close(fig)


def vis_attn_matrix(attn_matrix, file_idx, save_dir):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.matshow(attn_matrix)
    # print(attn_matrix)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fig.savefig(os.path.join(save_dir, file_idx))

    plt.close(fig)


def vis_joint_trail(img_trail, vedio_trail, vedio_length, save_dir):
    x = [i for i in range(vedio_length)]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(42):
        fig = plt.figure()
        for j in range(3):
            plt.subplot(3,1,j+1)
            plt.plot(x, vedio_trail[:, i, j], color='r', linestyle='-', alpha=0.8)
            plt.plot(x, img_trail[:, i, j], color='g', linestyle='-', alpha=0.6)


        fig.savefig(os.path.join(save_dir, str(i) + '.jpg'),dpi=200)
        plt.close(fig)


def vis_latent_vector(latent_vector, save_dir, file_name, bbox_idx):
    latent_vector = rearrange(latent_vector, '(c n) w h -> c (w n) h', n = 16).cpu().numpy()
    if not os.path.exists(os.path.join(save_dir, file_name + '_' + bbox_idx)):
        os.makedirs(os.path.join(save_dir, file_name + '_' + bbox_idx))
    for i in range(48):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(latent_vector[i])

        fig.savefig(os.path.join(save_dir, file_name + '_' + bbox_idx, str(i)))
        plt.close(fig)


def vis_joint_trail_gt(img_trail, vedio_trail, gt_trail, vedio_length, save_dir):
    x = [i for i in range(vedio_length)]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(42):
        fig = plt.figure(figsize=(16,8))
        for j in range(3):
            plt.subplot(3,1,j+1)
            plt.plot(x, vedio_trail[:, i, j], color='r', linestyle='-')
            plt.plot(x, img_trail[:, i, j], color='g', linestyle='--')
            plt.plot(x, gt_trail[:, i, j], color='orange', linestyle='--')

        fig.savefig(os.path.join(save_dir, str(i) + '.jpg'))
        plt.close(fig)




