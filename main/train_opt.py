import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
import sys
import torch.backends.cudnn as cudnn

sys.path.append('..')
from main.config import cfg
from common.base import TrainerOpt
from torch.utils.data import DataLoader
from models.model_opt import Model
from common.dataset_interhand import Dataset


def rearrange_input(batch_nof_hands, batch_imgs, batch_targets_singles, batch_targets_inters,
                    batch_targets_weights_singles, batch_targets_weights_inters):
    nof_hands = torch.sum(batch_nof_hands)
    input_imgs = torch.zeros((nof_hands, 3, cfg.input_img_shape[0], cfg.input_img_shape[1]))
    targets_singles = torch.zeros((nof_hands, cfg.joint_num, 3, int(cfg.output_hm_shape[0] * cfg.simdr_split_ratio)))
    targets_inters = torch.zeros((nof_hands, cfg.joint_num * 2, 3, int(cfg.output_hm_shape[0] * cfg.simdr_split_ratio)))
    targets_weights_singles = torch.zeros((nof_hands, cfg.joint_num))
    targets_weights_inters = torch.zeros((nof_hands, cfg.joint_num * 2))
    base_index = 0
    for i in range(0, batch_nof_hands.shape[0]):
        input_imgs[base_index: base_index + batch_nof_hands[i]] = batch_imgs[i][:batch_nof_hands[i]]
        targets_singles[base_index: base_index + batch_nof_hands[i]] = batch_targets_singles[i][:batch_nof_hands[i]]
        targets_inters[base_index: base_index + batch_nof_hands[i]] = batch_targets_inters[i][:batch_nof_hands[i]]
        targets_weights_singles[base_index: base_index + batch_nof_hands[i]] = batch_targets_weights_singles[i][:batch_nof_hands[i]]
        targets_weights_inters[base_index: base_index + batch_nof_hands[i]] = batch_targets_weights_inters[i][:batch_nof_hands[i]]
        base_index += batch_nof_hands[i]
    return input_imgs, targets_singles.cuda(), targets_inters.cuda(),targets_weights_singles.cuda(), targets_weights_inters.cuda()


def rearrange_output(batch_nof_hands, joint_simdr_single_out, joint_simdr_inter_out, batch_hand_type):
    batch_size = batch_nof_hands.shape[0]

    joint_type = {'right': np.arange(0, cfg.joint_num), 'left': np.arange(cfg.joint_num, cfg.joint_num * 2)}
    pred_simdrs = torch.zeros((batch_size, cfg.joint_num * 2, 3, int(cfg.output_hm_shape[0] * cfg.simdr_split_ratio))).cuda()
    base_index = 0
    for i in range(batch_size):
        if batch_hand_type[i] == 'right':
            pred_simdrs[i, joint_type['right']] = joint_simdr_single_out[base_index]
        elif batch_hand_type[i] == 'left':
            pred_simdrs[i, joint_type['left']] = joint_simdr_single_out[base_index]
        elif batch_hand_type[i] == 'interacting':
            pred_simdrs[i] = joint_simdr_inter_out[base_index]
        elif batch_hand_type[i] == 'two':
            pred_simdrs[i, joint_type['right']] = joint_simdr_single_out[base_index]
            pred_simdrs[i, joint_type['left']] = joint_simdr_single_out[base_index + 1]
        base_index += batch_nof_hands[i]
    return pred_simdrs


def simdr2coord(pred_simdrs, batch_size):

    simdr_x = pred_simdrs[:, :, 0, :]
    simdr_y = pred_simdrs[:, :, 1, :]
    simdr_z = pred_simdrs[:, :, 2, :]
    idx_x = torch.argmax(simdr_x, 2)
    idx_y = torch.argmax(simdr_y, 2)
    idx_z = torch.argmax(simdr_z, 2)
    joint_coord = torch.zeros((batch_size, 42, 3)).cuda()
    joint_coord[:, :, 0] = (idx_x - cfg.output_hm_shape[0]/2) /cfg.output_hm_shape[0]
    joint_coord[:, :, 1] = (idx_y- cfg.output_hm_shape[1]/2) /cfg.output_hm_shape[1]
    joint_coord[:, :, 2] = (idx_z - cfg.output_hm_shape[2]/2) /cfg.output_hm_shape[2]
    # print(joint_coord.shape)

    return joint_coord


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():

    opt = parse_opt()
    cfg.set_args(gpu_ids=opt.gpu_ids, continue_train=opt.continue_train)

    trainer = TrainerOpt()
    trainer._make_batch_generator()
    trainer._make_model()

    total_params = sum(p.numel() for p in trainer.model_opt.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in trainer.model_opt.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        running_loss = {}
        running_loss['loss_geo_right'] = 0
        running_loss['loss_geo_left'] = 0
        running_loss['loss_3d'] = 0
        for i, data in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            batch_pred_coord = data['pred_coord']
            # print(batch_pred_coord.shape)
            loss = trainer.model_opt(batch_pred_coord, data['gt_joint_img'], data['joint_valid'], data['gt_hand_type_array'], 'train')

            # print(data['gt_joint_img'][0])
            # print(batch_pred_coord[0])
            # return

            sum(loss[k] for k in loss).backward()

            for k, v in loss.items():
                running_loss[k] += v.detach()

            trainer.optimizer.step()
            trainer.gpu_timer.toc()

            if (i % 500 == 0):
                screen = [
                    'Epoch %d/%d itr %d:' % (epoch, cfg.end_epoch, i),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time,
                        trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
                screen += ['%s: %.4f' % (k, v / i) for k, v in running_loss.items()]
                trainer.logger.info(' '.join(screen))
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

            # lr_scheduler.step()

        screen = ['%s: %.4f' % ('running_loss_' + k, v / len(trainer.batch_generator)) for k, v in
                  running_loss.items()]
        trainer.logger.info(' '.join(screen))
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model_opt.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)

    return 0

            # preds['pred_simdrs'].append(pred_simdrs)
            # preds['inv_transs'].append(batch_inv_trans)


    #
    # # evaluate
    # preds = {k: np.concatenate(v) for k, v in preds.items()}
    # tester._evaluate(preds)


if __name__ == "__main__":
    main()