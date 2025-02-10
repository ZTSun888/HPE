import argparse
import os
import sys
import cv2

import numpy as np
import torch

sys.path.append('..')
from main.config import cfg
# from common.dataset_interhand import Dataset
# from torch.utils.data import DataLoader
# from models.model import get_model
from common.base import Trainer
# from einops import rearrange



def rearrange_input(batch_nof_hands, batch_imgs, batch_targets_singles, batch_targets_inters,
                    batch_targets_weights_singles, batch_targets_weights_inters,
                    batch_js_singles, batch_js_inters,
                    batch_js_valid_singles, batch_js_valid_inters):
    nof_hands = torch.sum(batch_nof_hands)
    input_imgs = torch.zeros((nof_hands, 3, cfg.input_img_shape[0], cfg.input_img_shape[1]))
    targets_singles = torch.zeros((nof_hands, cfg.joint_num, 3, cfg.output_hm_shape[0]))
    targets_inters = torch.zeros((nof_hands, cfg.joint_num * 2, 3, cfg.output_hm_shape[0]))
    targets_weights_singles = torch.zeros((nof_hands, cfg.joint_num))
    targets_weights_inters = torch.zeros((nof_hands, cfg.joint_num * 2))
    targets_js_singles = torch.zeros((nof_hands, cfg.joint_shift_num, 3))
    targets_js_inters = torch.zeros((nof_hands, cfg.joint_shift_num * 2, 3))
    targets_js_valid_singles = torch.zeros((nof_hands, cfg.joint_shift_num))
    targets_js_valid_inters = torch.zeros((nof_hands, cfg.joint_shift_num * 2))
    base_index = 0
    # print(batch_imgs.shape)
    for i in range(0, batch_nof_hands.shape[0]):
        # print()
        # print(str(i))
        # print('base_index: '+str(base_index))
        # print('batch_nof_hands[i]: '+ str(batch_nof_hands[i]))
        # print(' batch_imgs[i].shape: '+ str( batch_imgs[i].shape))
        input_imgs[base_index: base_index + batch_nof_hands[i]] = batch_imgs[i][:batch_nof_hands[i]]
        targets_singles[base_index: base_index + batch_nof_hands[i]] = batch_targets_singles[i][:batch_nof_hands[i]]
        targets_inters[base_index: base_index + batch_nof_hands[i]] = batch_targets_inters[i][:batch_nof_hands[i]]
        targets_weights_singles[base_index: base_index + batch_nof_hands[i]] = batch_targets_weights_singles[i][:batch_nof_hands[i]]
        targets_weights_inters[base_index: base_index + batch_nof_hands[i]] = batch_targets_weights_inters[i][:batch_nof_hands[i]]

        targets_js_singles[base_index: base_index + batch_nof_hands[i]] = batch_js_singles[i][:batch_nof_hands[i]]
        targets_js_inters[base_index: base_index + batch_nof_hands[i]] = batch_js_inters[i][:batch_nof_hands[i]]
        targets_js_valid_singles[base_index: base_index + batch_nof_hands[i]] = batch_js_valid_singles[i][:batch_nof_hands[i]]
        targets_js_valid_inters[base_index: base_index + batch_nof_hands[i]] = batch_js_valid_inters[i][:batch_nof_hands[i]]
        base_index += batch_nof_hands[i]

    # print(input_imgs.device)
    # print(targets_inters.device)
    return input_imgs, targets_singles.cuda(), targets_inters.cuda(), targets_weights_singles.cuda(), targets_weights_inters.cuda(),\
           targets_js_singles.cuda(), targets_js_inters.cuda(), targets_js_valid_singles.cuda(), targets_js_valid_inters.cuda()


def main():
    opt = parse_opt()
    # print(opt.gpu_ids)
    cfg.set_args(gpu_ids=opt.gpu_ids, continue_train=opt.continue_train)

    trainer = Trainer()
    trainer._make_model()
    trainer._make_batch_generator()


    # model_dict = trainer.model.state_dict()
    # print(model_dict.keys())

    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        running_loss = {}
        running_loss['simdr_single_2d'] = 0
        running_loss['simdr_inter_2d'] = 0
        running_loss['simdr_single_z'] = 0
        running_loss['simdr_inter_z'] = 0
        # running_loss['bone_map_single'] = 0
        # running_loss['bone_map_inter'] = 0

        for i, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            trainer.optimizer.zero_grad()

            input_img = inputs['img']
            targets_singles = targets['joint_simdr_singles']
            targets_inters = targets['joint_simdr_inters']
            targets_weights_singles = meta_info['joint_valid_singles']
            targets_weights_inters = meta_info['joint_valid_inters']


            # loss = trainer.model(input_img, targets_singles, targets_inters, targets_weights_singles, targets_weights_inters,
            #                      targets_js_singles, targets_js_inters, targets_js_valid_singles, targets_js_valid_inters,
            #                      'train')
            loss = trainer.model(input_img, targets_singles, targets_inters, targets_weights_singles,
                                 targets_weights_inters, 'train')


            sum(loss[k] for k in loss).backward()

            for k, v in loss.items():
                running_loss[k] += v.detach()

            trainer.optimizer.step()
            trainer.gpu_timer.toc()

            if (i % 1 == 0):
                screen = [
                    'Epoch %d/%d itr %d:' % (epoch, cfg.end_epoch, i),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time,
                        trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
                screen += ['%s: %.4f' % ('loss_' + k, v/i) for k, v in running_loss.items()]
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
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
        }, epoch)

        break

    return 0


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    opt = parser.parse_args()

    if not opt.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in opt.gpu_ids:
        gpus = opt.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        opt.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return opt


if __name__ == "__main__":
    main()