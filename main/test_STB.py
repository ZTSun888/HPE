import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
import sys
import torch.backends.cudnn as cudnn

sys.path.append('..')
from main.config import cfg
from common.base import TesterSTB


def simdr2coord(pred_simdrs):
    batch_size = pred_simdrs.shape[0]
    simdr_x = pred_simdrs[:, :, 0, :]
    simdr_y = pred_simdrs[:, :, 1, :]
    simdr_z = pred_simdrs[:, :, 2, :]
    idx_x = np.argmax(simdr_x, 2)
    idx_y = np.argmax(simdr_y, 2)
    idx_z = np.argmax(simdr_z, 2)
    joint_coord = np.zeros((batch_size, 21, 3))
    joint_coord[:, :, 0] = idx_x
    joint_coord[:, :, 1] = idx_y
    joint_coord[:, :, 2] = idx_z
    # print(joint_coord.shape)
    return joint_coord


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
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


    tester = TesterSTB(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()

    preds= {'pred_coord': [], 'inv_trans': []}
    with torch.no_grad():
        for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
            # forward
            # batch_nof_hands = meta_info['nof_hand']
            input_img = inputs['img']
            # batch_idx = meta_info['index']
            targets_singles = targets['joint_simdr_singles']
            targets_weights_singles = meta_info['joint_valid_singles']
            batch_inv_trans = meta_info['inv_trans']
            out = tester.model(input_img, targets_singles,targets_weights_singles,
                               'test')

            joint_simdr_single_out = out['joint_simdr_single'].cpu().numpy()

            joint_coord = simdr2coord(joint_simdr_single_out)
            # joint_coord = simdr2coord(targets_singles)
            preds['pred_coord'].append(joint_coord)
            preds['inv_trans'].append(batch_inv_trans)

    # evaluate
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    tester._evaluate(preds)


if __name__ == "__main__":
    main()