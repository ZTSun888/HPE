import os
import os.path as osp
import sys

class Config:

    # GPUS = (0)
    # num_gpus = len(GPUS.split(','))
    #
    input_img_shape = (224, 224)
    output_hm_shape = (448, 448, 448)
    # input_img_shape = (256, 256)
    # output_hm_shape = (64, 64, 64)
    bbox_3d_size = 400
    bbox_3d_size_root = 400  # depth axis
    output_root_hm_shape = 448  # depth axis
    joint_num = 21
    sigma = 4

    # # interhand2.6M model config
    # input_img_shape = (256, 256)
    # output_hm_shape = (64,64,64)  # (depth, height, width)
    # sigma = 2.5
    # bbox_3d_size = 400  # depth axis
    # bbox_3d_size_root = 400  # depth axis
    # output_root_hm_shape = 64  # depth axis
    # # output_root_hm_shape = 448  # depth axis
    # joint_num = 21

    # train_lr_step = [5,35]
    train_lr_step = [15,17]
    # lr = 1e-2
    lr = 1e-4
    lr_factor = 10
    lr_decay = 0.95
    train_batch_size = 32
    end_epoch = 80
    # max_peaks_num = 3

    vedio_pad = 16
    # smoothnet windows pad
    # vedio_pad = 32

    ## testing config
    test_batch_size = 32

    # simdr_split_ratio = 1.0

    gpu_ids = '0'
    num_gpus = 1
    continue_train = False

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, '..')
    output_dir = os.path.join(root_dir, 'output')
    model_dir = os.path.join(output_dir, 'model_dump')
    model_opt_dir = os.path.join(output_dir, 'model_opt_dump')
    model_inter_dir = os.path.join(output_dir, 'model_dump_inter')
    model_vedio_dir = os.path.join(output_dir, 'model_dump_vedio')
    model_vedio_STB_dir = os.path.join(output_dir, 'model_dump_vedio(STB)')
    model_peak_dir = os.path.join(output_dir, 'model_dump_peak')
    model_discriminator_dir = os.path.join(output_dir, 'model_dump_dis')
    model_vedio_dir_smoothnet = os.path.join(output_dir, 'model_dump_video_smoothnet')
    model_dir_Ego3D = os.path.join(output_dir, 'model_dump(Ego3D)')
    log_dir = osp.join(output_dir, 'log')
    joint_shift_num = 35

    # bone_num = 20
    # bone_map_downscale = 8
    # bone_index = [[0, 1], [1, 2], [2, 3], [3, 20],
    #                [4, 5], [5, 6], [6, 7], [7, 20],
    #                [8, 9], [9, 10], [10, 11], [11, 20],
    #                [12, 13], [13, 14], [14, 15], [15, 20],
    #                [16, 17], [17, 18], [18, 19], [19, 20]]
    #               # [21,22], [22,23], [23,24], [24,41],
    #               # [25,26], [26,27], [27,28], [28,41],
    #               # [29,30], [30,31], [31,32], [32,41],
    #               # [33,34], [34,35], [35,36], [36,41],
    #               # [37,38], [38,39], [39,40], [40,41]]

    # for STB joint idx
    # shift_index = [[2, 0], [3, 0], [4, 0],
    #                [6, 0], [7, 0], [8, 0],
    #                [10, 0], [11, 0], [12, 0],
    #                [14, 0], [15, 0], [16, 0],
    #                [18, 0], [19, 0], [20, 0],
    #                [0, 1], [1, 2], [2, 3], [3, 4],
    #                [0, 5], [5, 6], [6, 7], [7, 8],
    #                [0, 9], [9, 10], [10, 11], [11, 12],
    #                [0, 13], [13, 14], [14, 15], [15, 16],
    #                [0, 17], [17, 18], [18, 19], [19, 20]]
    # acc_loss_weight = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5,
    #                    1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5]
    bone_num = 20
    bone_index = [[0, 1], [1, 2], [2, 3], [3, 4],
                  [0, 5], [5, 6], [6, 7], [7, 8],
                  [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16],
                  [0, 17], [17, 18], [18, 19], [19, 20]]
                  # [21, 22], [22, 23], [23, 24], [24, 25],
                  # [21, 26], [26, 27], [27, 28], [28, 29],
                  # [21, 30], [30, 31], [31, 32], [32, 33],
                  # [21, 34], [34, 35], [35, 36], [36, 37],
                  # [21, 38], [38, 39], [39, 40], [40, 41]]

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()