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

class _Optblock(nn.Module):
    def __init__(self, joint_num, bone_num):
        super(_Optblock, self).__init__()

        self.joint_num = joint_num
        self.bone_num = bone_num

        self.simdr2coord = make_linear_layers([cfg.output_hm_shape[0], 1024, 128], relu_final=False)
        self.feature_mlp = make_linear_layers([49, 256, 128], relu_final=False)

        self.feature2bone_conv = make_conv_layers([self.joint_num*2 + 768, 1024, self.bone_num], bnrelu_final=False)
        self.feature2bone_lin = make_linear_layers([128, 32, 3], relu_final=False)
        self.dim_qk = 128
        self.scale = self.dim_qk ** -0.5
        self.to_q = nn.Linear(3, 128, bias=False)
        self.to_k = nn.Linear(3, 128, bias=False)
        self.to_v = nn.Linear(3, 128, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        # self.get_bone_feature = nn.Linear(128, 3136, bias=False)

        self.feature2bone_vec = nn.Linear(128, 3)
        # self.bone2feature = nn.Linear(3, 128)
        self.z_conv = make_conv_layers([self.joint_num*2 + self.bone_num, 256, self.joint_num], bnrelu_final=False)
        self.mlp_head_z = make_linear_layers([128, 256, cfg.output_hm_shape[2]], relu_final=False)


    def forward(self, pred_2d, feature):
        feature = rearrange(feature, 'b c h w -> b c (h w)')
        feature = self.feature_mlp(feature)
        pred_2d = rearrange(pred_2d, 'b j n h -> b (j n) h')
        pred_2d_feature = self.simdr2coord(pred_2d)
        feature = torch.cat((feature, pred_2d_feature), dim=1)
        bone_feature = self.feature2bone_conv(feature)
        bone_vec = self.feature2bone_lin(bone_feature)
        q = self.to_q(bone_vec)
        q = q * self.scale
        k = self.to_k(bone_vec)
        v = self.to_v(bone_vec)
        attn = torch.matmul(q, k.transpose(1,2))
        # attn_wosoftmax = attn.detach()
        attn = self.softmax(attn)
        attn_out = torch.matmul(attn, v)
        bone_vec = self.feature2bone_vec(attn_out)

        # bone_feature = self.bone2feature(bone_vec)
        final_feature = torch.cat((attn_out, pred_2d_feature), dim=1)
        pred_z1 = self.z_conv(final_feature) #[b, joint_num, 128]
        pred_z1 = self.mlp_head_z(pred_z1)

        # pred_z = pred_z1 * 0.5 + pred_z2 * 0.5
        pred_z = pred_z1
        return pred_z, bone_vec#, attn_wosoftmax


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




class PoseNet(nn.Module):
    def __init__(self, joint_num, joint_bone_num):
        super(PoseNet, self).__init__()

        self.joint_num = joint_num
        self.joint_bone_num = joint_bone_num
        # self.simdr_split_ratio = cfg.simdr_split_ratio
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

        # self.feature_linear = nn.Linear(49, cfg.output_hm_shape[2])
        # self.con1d_z_1 = nn.Sequential(nn.Conv1d(in_channels=768 + 2 * self.joint_num,
        #                                          out_channels=256,
        #                                          kernel_size=1,
        #                                          stride=1,
        #                                          padding=0
        #                                          ),
        #                                nn.BatchNorm1d(256),
        #                                nn.GELU())
        # self.con1d_z_2 = nn.Conv1d(in_channels=256,
        #                            out_channels=self.joint_num,
        #                            kernel_size=1,
        #                            stride=1,
        #                            padding=0
        #                            )
        self.opt_block = _Optblock(self.joint_num, self.joint_bone_num)
        # self.joint_shift_num = 15

        # self.js_linear = make_linear_layers([cfg.output_hm_shape[0] * 3, 1024, 3], relu_final=False)
        # self.js_shrink = nn.Conv1d(self.joint_num, joint_shift_num, 1)

        self.apply(self._init_weights)


    def forward(self, feature):
        x = self.deconv_layers(feature)
        # feature_vector = x
        x = self.final_layer(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        pred_2d = torch.stack((pred_x, pred_y), dim=2)

        pred_z, bone_dir = self.opt_block(pred_2d.detach(), feature)

        return pred_2d, pred_z, bone_dir#, attn


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
