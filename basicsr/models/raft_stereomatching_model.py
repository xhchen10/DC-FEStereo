import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .stereomatching_model import StereoMatchingModel


@MODEL_REGISTRY.register()
class RAFTStereoMatchingModel(StereoMatchingModel):
    def feed_data(self, data):
        self.left = data['frame']['left'].to(self.device)
        self.right = data['frame']['right'].to(self.device)
        if 'disparity_gt' in data:
            self.gt = data['disparity_gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.left, self.right)

        l_total = 0
        loss_dict = OrderedDict()

        gt_disp = self.gt.unsqueeze(1)
        mask = (gt_disp > 0)
        mask.detach_()

        n_predictions = len(self.output)
        l_disp_total = 0.
        loss_gamma = 0.9
        for i in range(n_predictions):
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            l_disp = F.smooth_l1_loss(self.output[i][mask], gt_disp[mask], size_average=True)
            l_disp_total += i_weight * l_disp

        l_total += l_disp_total
        loss_dict['l_disp'] = l_disp_total

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.left, self.right)[-1]
            self.output = self.output.squeeze(1)
        self.net_g.train()


@MODEL_REGISTRY.register()
class IERAFTStereoMatchingModel(RAFTStereoMatchingModel):
    def feed_data(self, data):
        self.left = data['frame']['left'].to(self.device)
        self.right = data['representation']['right'].to(self.device)
        if 'disparity_gt' in data:
            self.gt = data['disparity_gt'].to(self.device)


@MODEL_REGISTRY.register()
class EERAFTStereoMatchingModel(RAFTStereoMatchingModel):
    def feed_data(self, data):
        self.left = data['representation']['left'].to(self.device)
        self.right = data['representation']['right'].to(self.device)
        if 'disparity_gt' in data:
            self.gt = data['disparity_gt'].to(self.device)