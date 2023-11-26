import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import itertools

from basicsr.archs.practical_deep_stereo import embedding
from basicsr.archs.practical_deep_stereo import estimator
from basicsr.archs.practical_deep_stereo import matching
from basicsr.archs.practical_deep_stereo import regularization
from basicsr.archs.practical_deep_stereo import size_adapter
from basicsr.utils.registry import ARCH_REGISTRY

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1, h, w)).type_as(depth)  # [1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1, h, w)).type_as(depth)  # [1, H, W]
    ones = Variable(torch.ones(1, h, w)).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert (all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected),
                                                                              list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or (pixel_coords.shape[-2:] != depth.shape[-2:]):
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).contiguous().view(b, 3,
                                                                                           -1).cuda()  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2 * (X / Z) / (w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(b, h, w, 2)


def inverse_warp(feat, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        feat: the source feature (where to sample pixels) -- [B, CH, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B34')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')

    assert (intrinsics_inv.size() == intrinsics.size())

    batch_size, _, feat_height, feat_width = feat.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)

    pose_mat = pose.cuda()

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:],
                                 padding_mode)  # [B,H,W,2]
    projected_feat = torch.nn.functional.grid_sample(feat, src_pixel_coords, padding_mode=padding_mode,
                                                     align_corners=True)

    return projected_feat


@ARCH_REGISTRY.register()
class MVSRecurNet(nn.Module):
    def __init__(self, max_disp):
        super(MVSRecurNet, self).__init__()

        self._size_adapter = size_adapter.SizeAdapter()
        self._embedding_img = embedding.Embedding(number_of_input_features=3)
        self._embedding_event = embedding.Embedding(number_of_input_features=15, shortcut=False)
        self._matching_op_event_img = matching.MatchingOperation()
        self._matching_op_imgs = matching.MatchingOperation()
        self._matching_op_events = matching.MatchingOperation()
        self._regularization_event_img = regularization.Regularization(number_of_features=8, overall=False)
        self._regularization_imgs = regularization.Regularization(number_of_features=8, overall=False)
        self._regularization_events = regularization.Regularization(number_of_features=8, overall=False)
        self._regularization_final = regularization.Regularization(number_of_features=8)
        self._estimator = estimator.SubpixelMap()

        # for param in itertools.chain(self._embedding_img.parameters(),
        #                              self._embedding_event.parameters(),
        #                              self._matching_op_event_img.parameters(),
        #                              self._matching_op_imgs.parameters(),
        #                              self._matching_op_events.parameters()):
        #     param.requires_grad = False

        if (max_disp + 1) % 32 != 0:
            raise ValueError(
                '"maximum_disparity" + 1 should be multiple of 32, e.g.,'
                '"maximum disparity" can be equal to 31, 63, 95...')
        self.max_disp = max_disp
        # During the embedding spatial dimensions of an input are downsampled
        # 4x times. Therefore, "maximum_disparity" of matching module is
        # computed as (maximum_disparity + 1) / 4 - 1.
        self.max_disp_fea = (max_disp + 1) // 4 - 1

    def forward(self, imgs, events,
                intrinsic, baseline_mul_focal_length,
                poses_ev_img, poses_img0_img1, poses_ev0_img1):
        """
        img_0, event_0, event_1 -> target
        img_1 -> ref
        Estimate the disparity of ref (img_1, left)
        """
        # with torch.no_grad():
        if True:
            img_fea_list, shortcut_list, event_fea_list = [], [], []
            for img, event in zip(imgs, events):
                img_fea, shortcut = self._embedding_img(self._size_adapter.pad(img))
                img_fea_list.append(img_fea), shortcut_list.append(shortcut)
                event_fea = self._embedding_event(self._size_adapter.pad(event))
                event_fea_list.append(event_fea)

            intrinsic_inv = torch.inverse(intrinsic)
            intrinsic4 = intrinsic.clone()
            intrinsic_inv4 = intrinsic_inv.clone()
            intrinsic4[:, :2, :] = intrinsic4[:, :2, :] / 4
            intrinsic_inv4[:, :2, :2] = intrinsic_inv4[:, :2, :2] * 4
            baseline_mul_focal_length = baseline_mul_focal_length.view(-1, 1, 1)

        lstm_state_bottom_event_img, lstm_state_bottom_imgs, lstm_state_bottom_events, lstm_state_bottom_final = None, None, None, None

        outputs = []

        seq_len = len(imgs) - 1
        for i in range(seq_len):
            img_0_fea, img_1_fea = img_fea_list[i], img_fea_list[i+1]
            event_0_fea, event_1_fea = event_fea_list[i], event_fea_list[i+1]
            img_1_fea_shortcut = shortcut_list[i+1]
            pose_img0_img1, pose_ev0_img1, pose_ev_img = poses_img0_img1[i], poses_ev0_img1[i], poses_ev_img[i]

            b, c, h, w = img_0_fea.shape
            ones_vec = Variable(torch.ones(b, h, w)).cuda()

            # with torch.no_grad():
            if True:
                event_1_padded_fea = nn.ZeroPad2d((self.max_disp_fea, 0, 0, 0))(event_1_fea)
                cost_img_event, cost_imgs, cost_events = [], [], []
                for disparity in range(0, self.max_disp_fea + 1):
                    if disparity == 0:
                        concatenated_fea = torch.cat([img_1_fea, event_1_fea], dim=1)
                        matching_fea = self._matching_op_event_img(concatenated_fea)
                        cost_img_event.append(matching_fea)
                        # disparity = 0 is meaningless for depth
                        cost_imgs.append(torch.zeros_like(matching_fea))
                        cost_events.append(torch.zeros_like(matching_fea))
                    else:
                        event_1_shifted_fea = event_1_padded_fea[:, :, :, self.max_disp_fea - disparity:-disparity]
                        concatenated_fea = torch.cat([img_1_fea, event_1_shifted_fea], dim=1)
                        cost_img_event.append(self._matching_op_event_img(concatenated_fea))
                        # %% construct the cost volume of SfM by disparity -> depth -> inverse (backward) warping
                        # Is it correct? Verified in AsymStereo: test_inverse_warping_4_MVSNet_downsample.py
                        depth = torch.div(baseline_mul_focal_length, ones_vec * disparity * 4)  # d = bf/d
                        # Img
                        # pose should be ref to target
                        img_0_warped_fea = inverse_warp(img_0_fea, depth, pose_img0_img1, intrinsic4, intrinsic_inv4)
                        concatenated_fea = torch.cat([img_1_fea, img_0_warped_fea], dim=1)
                        cost_imgs.append(self._matching_op_imgs(concatenated_fea))
                        # Event
                        event_0_warped_fea = inverse_warp(event_0_fea, depth, pose_ev0_img1, intrinsic4, intrinsic_inv4)
                        event_1_warped_fea = inverse_warp(event_1_fea, depth, pose_ev_img, intrinsic4, intrinsic_inv4)
                        concatenated_fea = torch.cat([event_1_warped_fea, event_0_warped_fea], dim=1)
                        cost_events.append(self._matching_op_events(concatenated_fea))

                cost_img_event = torch.stack(cost_img_event, dim=2)
                cost_imgs = torch.stack(cost_imgs, dim=2)
                cost_events = torch.stack(cost_events, dim=2)

            cost_img_event, lstm_state_bottom_event_img = self._regularization_event_img(cost_img_event, img_1_fea_shortcut, lstm_state_bottom_event_img)
            cost_imgs, lstm_state_bottom_imgs = self._regularization_imgs(cost_imgs, img_1_fea_shortcut, lstm_state_bottom_imgs)
            cost_events, lstm_state_bottom_events = self._regularization_events(cost_events, img_1_fea_shortcut, lstm_state_bottom_events)

            costs = (cost_img_event + cost_imgs + cost_events) / 3
            costs, lstm_state_bottom_final = self._regularization_final(costs, img_1_fea_shortcut, lstm_state_bottom_final)

            if not self.training:
                outputs.append(self._size_adapter.unpad(self._estimator(costs)))
            else:
                outputs.append(self._size_adapter.unpad(costs))

        return outputs

    # def get_trainable_params(self):
    #     return itertools.chain(self._regularization_imgs.parameters(),
    #                            self._regularization_events.parameters(),
    #                            self._regularization_event_img.parameters(),
    #                            self._regularization_final.parameters())


if __name__ == '__main__':
    from thop import profile

    imgs = [torch.rand(1, 3, 256, 256).cuda(), torch.rand(1, 3, 256, 256).cuda(),
            torch.rand(1, 3, 256, 256).cuda(), torch.rand(1, 3, 256, 256).cuda()]
    events = [torch.rand(1, 15, 256, 256).cuda(), torch.rand(1, 15, 256, 256).cuda(),
              torch.rand(1, 15, 256, 256).cuda(), torch.rand(1, 15, 256, 256).cuda()]
    intrinsic = torch.rand(1, 3, 3).cuda()
    baseline_mul_focal_length = torch.rand(1).cuda()

    poses_ev_img = [torch.rand(1, 3, 4).cuda(), torch.rand(1, 3, 4).cuda(), torch.rand(1, 3, 4).cuda()]
    poses_img0_img1 = [torch.rand(1, 3, 4).cuda(), torch.rand(1, 3, 4).cuda(), torch.rand(1, 3, 4).cuda()]
    poses_ev0_img1 = [torch.rand(1, 3, 4).cuda(), torch.rand(1, 3, 4).cuda(), torch.rand(1, 3, 4).cuda()]

    model = MVSRecurNet(max_disp=95).cuda()
    model.eval()

    with torch.no_grad():
        flops, params = profile(model, inputs=(imgs, events,
                intrinsic, baseline_mul_focal_length,
                poses_ev_img, poses_img0_img1, poses_ev0_img1))
    print(flops, params)
