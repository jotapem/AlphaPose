import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import numpy as np

from libs.AlphaPose.opt import opt
from libs.AlphaPose.SPPE.src.utils.img import flip, shuffleLR
from libs.AlphaPose.SPPE.src.models.FastPose import createModel

import visdom
import time
import sys

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

class InferenNet_fast(nn.Module):
    def __init__(self, kernel_size, dataset, use_gpu=True):
        super(InferenNet_fast, self).__init__()

        model = createModel().cuda() if use_gpu else createModel().cpu()
        device = torch.device('cuda' if use_gpu else 'cpu')
        model.load_state_dict(torch.load('data/checkpoints/pose/sppe/duc_se.pth', map_location=device))
        model.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        return out
