import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
import sys
import torch.backends.cudnn as cudnn

sys.path.append('..')
from main.config import cfg
from common.base import TesterVedio
import scipy.signal as signal
from common.dataset_vedio_offline import DatasetVedioOffline


def main():
    mode = 'test'
    print("Creating test dataset...")
    testset = DatasetVedioOffline(mode, 'human_annot', cfg.vedio_pad)

    preds = {'pred_coord': []}
    video_num = testset.len()
    window_size = cfg.vedio_pad
    polyorder = 3
    if window_size % 2 == 0:
        window_size = window_size - 1
    for idx in tqdm(range(video_num)):
        inputs,video_len = testset.getitem(idx)
        opt_video = np.zeros_like(inputs)
        for j in range(42):
            for i in range(3):
                opt_video[:, j, i] = signal.savgol_filter(inputs[:, j, i], window_size, polyorder, axis=0)


        # print(opt_video.shape)
        preds['pred_coord'].append(opt_video)

    # evaluate
    # preds = {k: np.concatenate(v) for k, v in preds.items()}
    testset.evaluate(preds)


if __name__ == "__main__":
    main()