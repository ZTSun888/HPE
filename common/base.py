import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

from main.config import cfg
# from dataset import Dataset
from common.dataset_interhand import Dataset
from common.dataset_vedio import DatasetVedio
from common.dataset_stb import Dataset_STB
from create_gaussian_json.dataset_peak import Dataset_peak
from create_peak_json.dataset_discriminator import Dataset_discriminator
from common.dataset_stb_vedio import DatasetVedioSTB
from common.dataset_vedio_smoothnet import DatasetVedio_Smoothnet
from common.dataset_Ego3D import Dataset_Ego3D

from common.timer import Timer
from common.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from models.model import get_model
# from models.model_bone import get_model
from models_inter.model import get_model_inter
from models_vedio.model import get_model_vedio
from models.model_stb import get_model_stb
from models_vedio.model_smoothnet import get_model_vedio_smoothnet



class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):

    def __init__(self):
        super(Trainer, self).__init__(log_name='train_logs.txt')

    def get_optimizer(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        return optimizer

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr


    def set_lr(self, epoch):
        if len(cfg.train_lr_step) == 0:
            return cfg.lr

        for e in cfg.train_lr_step:
            if epoch < e:
                break
        if epoch < cfg.train_lr_step[-1]:
            idx = cfg.train_lr_step.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_factor ** len(cfg.train_lr_step))


    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset_loader = Dataset(cfg, 'train', 'human_annot')
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.train_batch_size * cfg.num_gpus,
                                 shuffle=False, pin_memory=True, num_workers=1)

        self.joint_num = trainset_loader.joint_num
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model()
        model = DataParallel(model).cuda()
        # model = model.cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
            last_epoch = start_epoch
        else:
            start_epoch = 0
            last_epoch = -1
        model.train()

        self.start_epoch = start_epoch
        self.last_epoch = last_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        model_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        start_epoch = ckpt['epoch'] + 1

        model.load_state_dict(ckpt['network'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])


        return start_epoch, model, optimizer


class Tester(Base):

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating test dataset...")
        testset_loader = Dataset(cfg, 'test', 'human_annot')
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=False, pin_memory=True)

        self.joint_num = testset_loader.joint_num
        self.batch_generator = batch_generator
        self.testset = testset_loader

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model()
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, preds):
        self.testset.evaluate(preds)


class TrainerVedio(Base):

    def __init__(self):
        super(TrainerVedio, self).__init__(log_name='train_vedio_logs.txt')

        self.lr = cfg.lr

    def get_optimizer(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        return optimizer

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr


    def set_lr(self, epoch):
        if len(cfg.train_lr_step) == 0:
            return cfg.lr
        for e in cfg.train_lr_step:
            if epoch < e:
                break
        if epoch < cfg.train_lr_step[-1]:
            idx = cfg.train_lr_step.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_factor ** len(cfg.train_lr_step))

        # self.lr *= cfg.lr_decay
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] *= cfg.lr_decay


    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset_loader = DatasetVedio(cfg, 'train', 'human_annot', cfg.vedio_pad)
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.train_batch_size * cfg.num_gpus,
                                 shuffle=True, pin_memory=True)

        self.joint_num = trainset_loader.joint_num
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph...")

        model = get_model_vedio()
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
            last_epoch = start_epoch
        else:
            start_epoch = 0
            last_epoch = -1

        self.start_epoch = start_epoch
        self.last_epoch = last_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_vedio_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_vedio_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        model_path = osp.join(cfg.model_vedio_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        start_epoch = ckpt['epoch'] + 1

        model.load_state_dict(ckpt['network'], strict=False)
        # optimizer.load_state_dict(ckpt['optimizer'])
        return start_epoch, model, optimizer


class TesterVedio(Base):

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(TesterVedio, self).__init__(log_name='test_vedio_logs.txt')

    def _make_batch_generator(self, inference):
        # data load and construct batch generator
        self.logger.info("Creating test dataset...")
        testset_loader = DatasetVedio(cfg, 'test', 'human_annot', cfg.vedio_pad, inference=inference)
        if not inference:
            batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.test_batch_size * cfg.num_gpus,
                                 shuffle=False, pin_memory=True)
        else:
            batch_generator = DataLoader(dataset=testset_loader, batch_size=1, shuffle=False, pin_memory=True)

        self.joint_num = testset_loader.joint_num
        self.itr_per_epoch = math.ceil(testset_loader.__len__() / cfg.num_gpus / cfg.test_batch_size)
        self.batch_generator = batch_generator
        self.testset = testset_loader

    def _make_model(self):
        # prepare network

        model = get_model_vedio()
        model = DataParallel(model).cuda()
        model_path = os.path.join(cfg.model_vedio_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, preds):
        self.testset.evaluate(preds)


class TrainerSTB(Base):

    def __init__(self):
        super(TrainerSTB, self).__init__(log_name='train_logs.txt')

    def get_optimizer(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        return optimizer

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr


    def set_lr(self, epoch):
        if len(cfg.train_lr_step) == 0:
            return cfg.lr

        for e in cfg.train_lr_step:
            if epoch < e:
                break
        if epoch < cfg.train_lr_step[-1]:
            idx = cfg.train_lr_step.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_factor ** len(cfg.train_lr_step))


    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset_loader = Dataset_STB(cfg, 'train')
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.train_batch_size * cfg.num_gpus,
                                 shuffle=True, pin_memory=True, num_workers=8)

        self.joint_num = trainset_loader.joint_num
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model_stb()
        model = DataParallel(model).cuda()
        # model = model.cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
            last_epoch = start_epoch
        else:
            start_epoch = 0
            last_epoch = -1
        model.train()

        self.start_epoch = start_epoch
        self.last_epoch = last_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        model_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        start_epoch = ckpt['epoch'] + 1

        model.load_state_dict(ckpt['network'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])


        return start_epoch, model, optimizer


class TesterSTB(Base):

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(TesterSTB, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating test dataset...")
        testset_loader = Dataset_STB(cfg, 'test')
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=False, pin_memory=True)

        self.joint_num = testset_loader.joint_num
        self.batch_generator = batch_generator
        self.testset = testset_loader

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model_stb()
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, preds):
        self.testset.evaluate(preds)


class TrainerVedio_STB(Base):

    def __init__(self):
        super(TrainerVedio_STB, self).__init__(log_name='train_vedio_logs.txt')

        self.lr = cfg.lr

    def get_optimizer(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        return optimizer

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr


    def set_lr(self, epoch):
        if len(cfg.train_lr_step) == 0:
            return cfg.lr
        for e in cfg.train_lr_step:
            if epoch < e:
                break
        if epoch < cfg.train_lr_step[-1]:
            idx = cfg.train_lr_step.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_factor ** len(cfg.train_lr_step))

        # self.lr *= cfg.lr_decay
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] *= cfg.lr_decay


    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset_loader = DatasetVedioSTB(cfg, 'train', cfg.vedio_pad)
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.train_batch_size * cfg.num_gpus,
                                 shuffle=True, pin_memory=True)

        self.joint_num = trainset_loader.joint_num
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph...")

        model = get_model_vedio()
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
            last_epoch = start_epoch
        else:
            start_epoch = 0
            last_epoch = -1

        self.start_epoch = start_epoch
        self.last_epoch = last_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_vedio_STB_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_vedio_STB_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        model_path = osp.join(cfg.model_vedio_STB_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        start_epoch = ckpt['epoch'] + 1

        model.load_state_dict(ckpt['network'], strict=False)
        # optimizer.load_state_dict(ckpt['optimizer'])
        return start_epoch, model, optimizer


class TesterVedio_STB(Base):

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(TesterVedio_STB, self).__init__(log_name='test_vedio_logs.txt')

    def _make_batch_generator(self, inference):
        # data load and construct batch generator
        self.logger.info("Creating test dataset...")
        testset_loader = DatasetVedioSTB(cfg, 'test', cfg.vedio_pad, inference=inference)
        if not inference:
            batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.test_batch_size * cfg.num_gpus,
                                 shuffle=False, pin_memory=True)
        else:
            batch_generator = DataLoader(dataset=testset_loader, batch_size=1, shuffle=False, pin_memory=True)

        self.joint_num = testset_loader.joint_num
        self.itr_per_epoch = math.ceil(testset_loader.__len__() / cfg.num_gpus / cfg.test_batch_size)
        self.batch_generator = batch_generator
        self.testset = testset_loader

    def _make_model(self):
        # prepare network

        model = get_model_vedio()
        model = DataParallel(model).cuda()
        model_path = os.path.join(cfg.model_vedio_STB_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, preds):
        self.testset.evaluate(preds)


class TrainerVedio_Smoothnet(Base):

    def __init__(self):
        super(TrainerVedio_Smoothnet, self).__init__(log_name='train_vedio_logs.txt')

        self.lr = cfg.lr

    def get_optimizer(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        return optimizer

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr


    def set_lr(self):
        self.lr *= 0.95
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= cfg.lr_decay


    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset_loader = DatasetVedio_Smoothnet(cfg, 'train', 'human_annot', cfg.vedio_pad)
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.train_batch_size * cfg.num_gpus,
                                 shuffle=True, pin_memory=True)

        self.joint_num = trainset_loader.joint_num
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph...")

        model = get_model_vedio_smoothnet()
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
            last_epoch = start_epoch
        else:
            start_epoch = 0
            last_epoch = -1

        self.start_epoch = start_epoch
        self.last_epoch = last_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_vedio_dir_smoothnet, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_vedio_dir_smoothnet, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        model_path = osp.join(cfg.model_vedio_dir_smoothnet, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        start_epoch = ckpt['epoch'] + 1

        model.load_state_dict(ckpt['network'], strict=False)
        # optimizer.load_state_dict(ckpt['optimizer'])
        return start_epoch, model, optimizer


class TesterVedio_Smoothnet(Base):

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(TesterVedio_Smoothnet, self).__init__(log_name='test_vedio_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating test dataset...")
        testset_loader = DatasetVedio_Smoothnet(cfg, 'test', 'human_annot', cfg.vedio_pad)

        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.test_batch_size * cfg.num_gpus,
                                 shuffle=False, pin_memory=True)


        self.joint_num = testset_loader.joint_num
        self.itr_per_epoch = math.ceil(testset_loader.__len__() / cfg.num_gpus / cfg.test_batch_size)
        self.batch_generator = batch_generator
        self.testset = testset_loader

    def _make_model(self):
        # prepare network

        model = get_model_vedio_smoothnet()
        model = DataParallel(model).cuda()
        model_path = os.path.join(cfg.model_vedio_dir_smoothnet, 'snapshot_%d.pth.tar' % self.test_epoch)
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, preds):
        self.testset.evaluate(preds)


class TrainerEgo3D(Base):

    def __init__(self):
        super(TrainerEgo3D, self).__init__(log_name='train_logs(Ego3D).txt')

    def get_optimizer(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        return optimizer

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr


    def set_lr(self, epoch):
        if len(cfg.train_lr_step) == 0:
            return cfg.lr

        for e in cfg.train_lr_step:
            if epoch < e:
                break
        if epoch < cfg.train_lr_step[-1]:
            idx = cfg.train_lr_step.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_factor ** len(cfg.train_lr_step))


    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset_loader = Dataset_Ego3D(cfg, 'train')
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.train_batch_size * cfg.num_gpus,
                                 shuffle=True, pin_memory=True, num_workers=8)

        self.joint_num = trainset_loader.joint_num
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model()
        model = DataParallel(model).cuda()
        # model = model.cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
            last_epoch = start_epoch
        else:
            start_epoch = 0
            last_epoch = -1
        model.train()

        self.start_epoch = start_epoch
        self.last_epoch = last_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir_Ego3D, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir_Ego3D, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        model_path = osp.join(cfg.model_dir_Ego3D, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        start_epoch = ckpt['epoch'] + 1

        model.load_state_dict(ckpt['network'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])


        return start_epoch, model, optimizer


class TesterEgo3D(Base):

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(TesterSTB, self).__init__(log_name='test_logs(Ego3D).txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating test dataset...")
        testset_loader = Dataset_Ego3D(cfg, 'test')
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=False, pin_memory=True)

        self.joint_num = testset_loader.joint_num
        self.batch_generator = batch_generator
        self.testset = testset_loader

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir_Ego3D, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model()
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, preds):
        self.testset.evaluate(preds)
