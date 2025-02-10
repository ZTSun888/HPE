import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
import sys
import torch.backends.cudnn as cudnn

sys.path.append('..')
from main.config import cfg
from common.base import Tester
from thop import profile, clever_format

# def rearrange_input(batch_nof_hands, batch_imgs, batch_targets_singles, batch_targets_inters,
#                     batch_targets_weights_singles, batch_targets_weights_inters,
#                     batch_js_singles, batch_js_inters, batch_js_valid_singles, batch_js_valid_inters):
def rearrange_input(batch_nof_hands, batch_imgs, batch_targets_singles,# batch_targets_inters,
                    batch_targets_weights_singles, #batch_targets_weights_inters,
                    batch_js_singles, #batch_js_inters,
                    batch_js_valid_singles):#, batch_js_valid_inters):
    nof_hands = torch.sum(batch_nof_hands)
    input_imgs = torch.zeros((nof_hands, 3, cfg.input_img_shape[0], cfg.input_img_shape[1]))
    targets_singles = torch.zeros((nof_hands, cfg.joint_num, 3, cfg.output_hm_shape[0]))
    # targets_inters = torch.zeros((nof_hands, cfg.joint_num * 2, 3, cfg.output_hm_shape[0]))
    targets_weights_singles = torch.zeros((nof_hands, cfg.joint_num))
    # targets_weights_inters = torch.zeros((nof_hands, cfg.joint_num * 2))
    targets_js_singles = torch.zeros((nof_hands, cfg.joint_shift_num, 3))
    # targets_js_inters = torch.zeros((nof_hands, cfg.joint_shift_num * 2, 3))
    targets_js_valid_singles = torch.zeros((nof_hands, cfg.joint_shift_num))
    # targets_js_valid_inters = torch.zeros((nof_hands, cfg.joint_shift_num * 2))
    # indxs = torch.zeros((nof_hands))
    base_index = 0
    # print(batch_imgs.shape)
    for i in range(0, batch_nof_hands.shape[0]):
        input_imgs[base_index: base_index + batch_nof_hands[i]] = batch_imgs[i][:batch_nof_hands[i]]
        targets_singles[base_index: base_index + batch_nof_hands[i]] = batch_targets_singles[i][:batch_nof_hands[i]]
        # targets_inters[base_index: base_index + batch_nof_hands[i]] = batch_targets_inters[i][:batch_nof_hands[i]]
        targets_weights_singles[base_index: base_index + batch_nof_hands[i]] = batch_targets_weights_singles[i][:batch_nof_hands[i]]
        # targets_weights_inters[base_index: base_index + batch_nof_hands[i]] = batch_targets_weights_inters[i][:batch_nof_hands[i]]

        targets_js_singles[base_index: base_index + batch_nof_hands[i]] = batch_js_singles[i][:batch_nof_hands[i]]
        # targets_js_inters[base_index: base_index + batch_nof_hands[i]] = batch_js_inters[i][:batch_nof_hands[i]]
        targets_js_valid_singles[base_index: base_index + batch_nof_hands[i]] = batch_js_valid_singles[i][:batch_nof_hands[i]]
        # targets_js_valid_inters[base_index: base_index + batch_nof_hands[i]] = batch_js_valid_inters[i][:batch_nof_hands[i]]
        # indxs[base_index: base_index + batch_nof_hands[i]] = batch_idxs[i]
        base_index += batch_nof_hands[i]

    # return input_imgs, targets_singles.cuda(), targets_inters.cuda(),targets_weights_singles.cuda(), targets_weights_inters.cuda(),\
    #        targets_js_singles.cuda(), targets_js_inters.cuda(), targets_js_valid_singles.cuda(), targets_js_valid_inters.cuda()
    return input_imgs, targets_singles.cuda(), targets_weights_singles.cuda(), targets_js_singles.cuda(), targets_js_valid_singles.cuda()


def rearrange_output(joint_simdr_single_out, joint_simdr_inter_out, batch_hand_type):
# def rearrange_output(joint_simdr_single_out, targets_singles,atch_hand_type):
    batch_size = joint_simdr_single_out.shape[0]
    joint_type = {'right': np.arange(0, cfg.joint_num), 'left': np.arange(cfg.joint_num, cfg.joint_num * 2)}
    # pred_simdr_singles = torch.zeros((batch_size, cfg.joint_num, 3, int(cfg.output_hm_shape[0] * cfg.simdr_split_ratio)))
    pred_simdrs = np.zeros((batch_size, cfg.joint_num * 2, 3, cfg.output_hm_shape[0]))
    # target_simdr_singles = torch.zeros((batch_size, cfg.joint_num, 3, int(cfg.output_hm_shape[0] * cfg.simdr_split_ratio)))
    # target_simdrs = np.zeros((batch_size, cfg.joint_num * 2, 3, cfg.output_hm_shape[0]))
    for i in range(batch_size):
        if batch_hand_type[i] == 'right':
            pred_simdrs[i, joint_type['right']] = joint_simdr_single_out[i]
            # target_simdrs[i, joint_type['right']] = targets_singles[base_index]
        elif batch_hand_type[i] == 'left':
            pred_simdrs[i, joint_type['left']] = joint_simdr_single_out[i]
            # target_simdrs[i, joint_type['left']] = targets_singles[base_index]
        elif batch_hand_type[i] == 'interacting':
            pred_simdrs[i] = joint_simdr_inter_out[i]
            # target_simdrs[i] = targets_inters[base_index]
        # elif batch_hand_type[i] == 'interacting':
        #     pred_simdrs[i, joint_type['right']] = joint_simdr_single_out[base_index]
        #     pred_simdrs[i, joint_type['left']] = joint_simdr_single_out[base_index + 1]
            # target_simdrs[i, joint_type['right']] = targets_singles[base_index]
            # target_simdrs[i, joint_type['left']] = targets_singles[base_index + 1]
    return pred_simdrs #, target_simdrs


def simdr2coord(pred_simdrs):
    batch_size = pred_simdrs.shape[0]
    simdr_x = pred_simdrs[:, :, 0, :]
    simdr_y = pred_simdrs[:, :, 1, :]
    simdr_z = pred_simdrs[:, :, 2, :]
    idx_x = np.argmax(simdr_x, 2)
    idx_y = np.argmax(simdr_y, 2)
    idx_z = np.argmax(simdr_z, 2)
    joint_coord = np.zeros((batch_size, 42, 3))
    joint_coord[:, :, 0] = idx_x
    joint_coord[:, :, 1] = idx_y
    joint_coord[:, :, 2] = idx_z
    # print(joint_coord.shape)
    return joint_coord


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    # parser.add_argument('--test_set', type=str, dest='test_set')
    # parser.add_argument('--annot_subset', type=str, dest='annot_subset')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args


def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True


    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()


    # preds = {'pred_coord': [], 'inv_trans': [], 'target_joint_simdr': [], 'pred_simdrs': [], 'inter_attn': []}
    preds= {'pred_coord': [], 'inv_trans': []}
    with torch.no_grad():
        for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
            # forward
            # batch_nof_hands = meta_info['nof_hand']
            input_img = inputs['img']
            # batch_idx = meta_info['index']
            targets_singles = targets['joint_simdr_singles']
            targets_inters = targets['joint_simdr_inters']
            targets_weights_singles = meta_info['joint_valid_singles']
            targets_weights_inters = meta_info['joint_valid_inters']
            batch_hand_type = targets['hand_type']
            batch_inv_trans = meta_info['inv_trans']
            out = tester.model(input_img, targets_singles, targets_inters, targets_weights_singles, targets_weights_inters,
                               'test')
            # out = tester.model(input_imgs, targets_singles, targets_weights_singles,
            #                    targets_js_singles, targets_js_valid_singles,
            #                    'test')
            joint_simdr_single_out = out['joint_simdr_single'].cpu().numpy()
            joint_simdr_inter_out = out['joint_simdr_inter'].cpu().numpy()

            pred_simdrs = rearrange_output(joint_simdr_single_out, joint_simdr_inter_out, batch_hand_type)
            # print(pred_simdrs.shape)
            joint_coord = simdr2coord(pred_simdrs)
            # target_coord = simdr2coord(targets_simdrs, batch_nof_hands.shape[0])
            preds['pred_coord'].append(joint_coord)
            preds['inv_trans'].append(batch_inv_trans)
            # preds['target_joint_simdr'].append(targets['joint_simdr'])
            # preds['pred_simdrs'].append(pred_simdrs)
            # preds['inter_attn'].append(inter_attn)

            # opt_simdr_single_out = out['opt_simdr_single'].cpu().numpy()
            # opt_simdr_inter_out = out['opt_simdr_inter'].cpu().numpy()
            # pred_simdrs_opt= rearrange_output(batch_nof_hands, opt_simdr_single_out, opt_simdr_inter_out,
            #                                targets_singles.cpu().numpy(),
            #                                targets_inters.cpu().numpy(), batch_hand_types)
            # pred_simdrs_opt = rearrange_output(batch_nof_hands, opt_simdr_single_out,
            #                                    targets_singles.cpu().numpy(),batch_hand_types)
            # # print(pred_simdrs.shape)
            # joint_coord_opt = simdr2coord(pred_simdrs_opt, batch_nof_hands.shape[0])
            # preds_opt['pred_coord'].append(joint_coord_opt)
            # preds_opt['inv_trans'].append(batch_inv_trans)




    # evaluate
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    tester._evaluate(preds)

    # preds_1 = {k: np.concatenate(v) for k, v in preds_opt.items()}
    # tester._evaluate(preds_1)


if __name__ == "__main__":
    main()