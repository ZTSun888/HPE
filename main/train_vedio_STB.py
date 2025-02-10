import argparse
import os
import sys
import cv2

import numpy as np
import torch

sys.path.append('..')
from main.config import cfg
from common.base import TrainerVedio_STB



def main():
    opt = parse_opt()
    # print(opt.gpu_ids)
    cfg.set_args(gpu_ids=opt.gpu_ids, continue_train=opt.continue_train)

    trainer = TrainerVedio_STB()
    trainer._make_model()
    trainer._make_batch_generator()



    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        running_loss = {}
        # running_loss['simdr_single_2d'] = 0
        # running_loss['simdr_inter_2d'] = 0

        # running_loss['loss_3d'] = 0
        # running_loss['loss_tempcons'] = 0
        # running_loss['loss_smoothnet'] = 0
        # running_loss['loss_temp'] = 0
        running_loss['loss_bonelen'] = 0


        for i, (inputs, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            trainer.optimizer.zero_grad()

            loss = trainer.model(inputs, meta_info, 'train')

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
                screen += ['%s: %.4f' % (k, v/i) for k, v in running_loss.items()]
                trainer.logger.info(' '.join(screen))
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        # lr_scheduler.step()

        screen = ['%s: %.4f' % ('running_' + k, v / len(trainer.batch_generator)) for k, v in
                      running_loss.items()]
        trainer.logger.info(' '.join(screen))
        trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
        }, epoch)


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