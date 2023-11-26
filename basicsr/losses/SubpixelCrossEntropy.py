# © All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import torch as th
from torch import nn
from torch.nn import functional

from basicsr.utils.registry import LOSS_REGISTRY


def _unnormalized_laplace_probability(value, location, diversity):
    return th.exp(-th.abs(location - value) / diversity) / (2 * diversity)


@LOSS_REGISTRY.register()
class SubpixelCrossEntropy(nn.Module):
    def __init__(self, diversity=1.0, disparity_step=2):
        """Returns SubpixelCrossEntropy object.

        Args:
            disparity_step: disparity difference between near-by
                       disparity indices in "similarities" tensor.
            diversity: diversity of the target Laplace distribution,
                       centered at the sub-pixel ground truth.
        """
        super(SubpixelCrossEntropy, self).__init__()
        self._diversity = diversity
        self._disparity_step = disparity_step

    def forward(self, similarities, ground_truth_disparities, weights=None):
        """Returns sub-pixel cross-entropy loss.

        Cross-entropy is computed as

        - sum_d log( P_predicted(d) ) x P_target(d)
          -------------------------------------------------
                            sum_d P_target(d)

        We need to normalize the cross-entropy by sum_d P_target(d),
        since the target distribution is not normalized.

        Args:
            ground_truth_disparities: Tensor with ground truth disparities with
                        indices [example_index, y, x]. The
                        disparity values are floats. The locations with unknown
                        disparities are filled with 0.
            similarities: Tensor with similarities with indices
                         [example_index, disparity_index, y, x].
            weights: Tensor with weights of individual locations.
        """
        maximum_disparity_index = similarities.size(1)
        known_ground_truth_disparity = ground_truth_disparities.data != 0
        log_P_predicted = functional.log_softmax(similarities, dim=1)
        sum_P_target = th.zeros(ground_truth_disparities.size())
        sum_P_target_x_log_P_predicted = th.zeros(
            ground_truth_disparities.size())
        if similarities.is_cuda:
            sum_P_target = sum_P_target.cuda()
            sum_P_target_x_log_P_predicted = \
                sum_P_target_x_log_P_predicted.cuda()
        for disparity_index in range(maximum_disparity_index):
            disparity = disparity_index * self._disparity_step
            P_target = _unnormalized_laplace_probability(
                value=disparity,
                location=ground_truth_disparities,
                diversity=self._diversity)
            sum_P_target += P_target
            sum_P_target_x_log_P_predicted += (
                log_P_predicted[:, disparity_index] * P_target)
        entropy = -sum_P_target_x_log_P_predicted[
            known_ground_truth_disparity] / sum_P_target[
                known_ground_truth_disparity]
        if weights is not None:
            weights_with_ground_truth = weights[known_ground_truth_disparity]
            return (weights_with_ground_truth * entropy).sum() / (
                weights_with_ground_truth.sum() + 1e-15)
        return entropy.mean()
