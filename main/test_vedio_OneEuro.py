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
from test_script.utiles.OneEuroFilter import OneEuroFilter
from common.dataset_vedio_offline import DatasetVedioOffline
import time


def main():
    mode = 'test'
    print("Creating test dataset...")
    testset = DatasetVedioOffline(mode, 'human_annot', cfg.vedio_pad)

    preds = {'pred_coord': []}
    video_num = testset.len()
    for idx in tqdm(range(video_num)):
        inputs,video_len = testset.getitem(idx)
        freq = 30
        f_list = []
        for j in range(42):
            f_list.append([])
            for idx in range(3):
                f = OneEuroFilter(freq=freq)
                f_list[j].append(f)

        timestamp = 0.0  # seconds
        i = 0
        opt_video = []
        for i in range(video_len):
            start_time = time.time()
            filtered = np.zeros((42, 3), dtype=np.float32)
            for j in range(42):
                for idx in range(3):
                    filtered[j, idx] = f_list[j][idx](inputs[i][j][idx], timestamp)

            timestamp += 1.0 / freq
            if len(opt_video) > cfg.vedio_pad:
                opt_video.append(filtered)
            else:
                opt_video.append(inputs[i])
            end_time = time.time()
            print(end_time - start_time)

        opt_video = np.array(opt_video)
        # print(opt_video.shape)
        preds['pred_coord'].append(opt_video)

    # evaluate
    # preds = {k: np.concatenate(v) for k, v in preds.items()}
    testset.evaluate(preds)


if __name__ == "__main__":
    main()