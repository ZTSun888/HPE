import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import os
from main.config import Config as cfg
from torch import nn, einsum



def make_linear_layers(feat_dims, relu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and relu_final):
            layers.append(nn.GELU())

    return nn.Sequential(*layers)


class SpacialOptimization(nn.Module):
    def __init__(self,
                 n_head=4,
                 d_qk=32,
                 d_v=32):
        super(SpacialOptimization, self).__init__()

        # self.scale = d_qk ** -0.5
        # self.n_head = n_head
        # self.to_q = nn.Linear(3, n_head*d_qk, bias=False)
        # self.to_k = nn.Linear(3, n_head*d_qk, bias=False)
        # self.to_v = nn.Linear(3, n_head*d_v, bias=False)
        #
        # self.softmax = nn.Softmax(dim=-1)
        #
        # self.fc_1 = nn.Linear(n_head * d_qk, 64, bias=False)

        self.ff = make_linear_layers([3, 1024, 64, 32, 3], relu_final=False)

        self.apply(self._init_weights)


    def forward(self, x, mask):

        # q = self.to_q(x)
        # k = self.to_k(x)
        # v = self.to_v(x)
        # # residual = q
        # q, k, v = map(lambda x: rearrange(x, 'B j (h d) -> B j h d', h=self.n_head), (q, k, v))
        # q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        #
        #
        # # if mask is not None:
        # #     mask = mask[None, None, :, :]
        #
        # q = q * self.scale
        # attn = torch.matmul(q, k.transpose(2, 3))
        # if mask is not None:
        #     attn = attn.masked_fill(mask == 0, -1e9)
        #
        #
        # attn = self.softmax(attn)
        # output = torch.matmul(attn, v)
        # output = rearrange(output.transpose(1, 2), 'B j h d -> B j (h d)')
        #
        # # output = self.fc_2(self.gelu(self.fc_1(output)))
        # output = self.fc_1(output)
        # output = self.batch_norm_1(output)

        output = self.ff(x)
        # output = self.layer_norm_2(output)

        return output


    def _init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            if (m.bias is not None):  # nn.Linear in Attention has no attr bias
                nn.init.constant_(m.bias, 0)
