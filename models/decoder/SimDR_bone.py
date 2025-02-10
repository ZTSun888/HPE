import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import os
from main.config import Config as cfg

BN_MOMENTUM = 0.1

def make_linear_layers(feat_dims, relu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and relu_final):
            layers.append(nn.GELU())

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=1, stride=1, padding=0, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
            ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
            layers.append(nn.GELU())

    return nn.Sequential(*layers)


def make_conv2d_layers(feat_dims, kernel=3, stride=2, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.GELU())

    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.GELU(inplace=True))

    return nn.Sequential(*layers)

class PoseNet_z(nn.Module):
    def __init__(self, joint_num, bone_num):
        super(PoseNet_z, self).__init__()

        self.joint_num = joint_num
        self.bone_num = bone_num

        self.xy_downsample = nn.MaxPool2d(kernel_size=8)
        self.final_layer = make_conv_layers([self.joint_num + 256 + self.bone_num, 128, self.joint_num],  bnrelu_final=True)
        self.mlp_head_x = nn.Linear(3136, cfg.output_hm_shape[0])
        self.mlp_head_y = nn.Linear(3136, cfg.output_hm_shape[1])
        self.mlp_head_z = nn.Linear(3136, cfg.output_hm_shape[2])

        self.apply(self._init_weights)


    def forward(self, pred_xy, bone_map ,feature):
        pred_xy = self.xy_downsample(pred_xy)
        feature_cat = torch.cat((feature, pred_xy, bone_map), dim=1)
        pred = self.final_layer(feature_cat)
        pred_x = self.mlp_head_x(pred)
        pred_y = self.mlp_head_y(pred)
        pred_z = self.mlp_head_z(pred)

        pred_3d = torch.stack((pred_x, pred_y, pred_z), dim=2)
        return pred_3d


    def _init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)


class PoseNet(nn.Module):
    def __init__(self, joint_num, bone_num):
        super(PoseNet, self).__init__()

        self.joint_num = joint_num
        self.bone_num = bone_num
        self.inplanes = 768

        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256,256,256],
            num_kernels=[4,4,4])

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=self.joint_num,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.mlp_head_x = nn.Linear(3136, cfg.output_hm_shape[0])
        self.mlp_head_y = nn.Linear(3136, cfg.output_hm_shape[1])

        self.bone_conv = make_conv2d_layers([256, self.bone_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

        self.apply(self._init_weights)


    def forward(self, feature):
        x = self.deconv_layers(feature)
        feature_vector = x
        x = self.final_layer(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        pred_2d = torch.stack((pred_x, pred_y), dim=2)
        pred_xy = torch.matmul(pred_x[:, :, :, None], pred_y[:, :, None, :])

        bone_map = self.bone_conv(feature_vector)

        return pred_2d, bone_map, feature_vector, pred_xy


    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding


    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            # layers.append(nn.ReLU(inplace=True))
            layers.append(nn.GELU())
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)
            # if self.deconv_with_bias:
            #     nn.init.constant_(m.bias, 0)
