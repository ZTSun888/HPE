import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
import sys
import torch.backends.cudnn as cudnn

sys.path.append('..')
from main.config import cfg
from common.base import TesterVedio_STB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--inference', type=bool, default=False)
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


    tester = TesterVedio_STB(args.test_epoch)
    tester._make_batch_generator(args.inference)
    tester._make_model()

    preds = {'pred_coord': []}
    with torch.no_grad():
        for itr, (inputs, meta_info) in enumerate(tqdm(tester.batch_generator)):
            # forward
            # print(inputs)
            out = tester.model(inputs, meta_info, 'test').cpu().numpy()
            if args.inference:
                tester.testset.append_opt_coord_list(meta_info['vedio_idx'], out)

            preds['pred_coord'].append(out)
            # preds['pred_coord'].append(inputs[:, -1])
            # break
    # evaluate
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    tester._evaluate(preds)


if __name__ == "__main__":
    main()