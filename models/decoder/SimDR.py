import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import os
from main.config import Config as cfg

BN_MOMENTUM = 0.1

class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()

        self.joint_num = joint_num  # single hand
        self.simdr_split_ratio = cfg.simdr_split_ratio
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

        self.mlp_head_x = nn.Linear(3136, int(cfg.output_hm_shape[0] * self.simdr_split_ratio))
        self.mlp_head_y = nn.Linear(3136, int(cfg.output_hm_shape[1] * self.simdr_split_ratio))
        self.mlp_head_z = nn.Linear(3136, int(cfg.output_hm_shape[2] * self.simdr_split_ratio))
        self.apply(self._init_weights)


    def forward(self, x):
        print('simdr: '+str(x.shape))

        x = self.deconv_layers(x)
        x = self.final_layer(x)
        # print('simdr: ' + str(x.shape))

        x = rearrange(x, 'b c h w -> b c (h w)')
        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)
        pred_z = self.mlp_head_y(x)
        pred = torch.stack((pred_x,pred_y, pred_z), dim=2)
        return pred


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