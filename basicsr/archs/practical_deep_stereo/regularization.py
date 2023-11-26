# Copirights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import torch
from torch import nn

from . import network_blocks


class ContractionBlock3d(nn.Module):
    """Contraction block, that downsamples the input.

    The contraction blocks constitute the contraction part of
    the regularization network. Each block consists of 2x
    "donwsampling" convolution followed by conventional "smoothing"
    convolution.
    """

    def __init__(self, number_of_features):
        super(ContractionBlock3d, self).__init__()
        self._downsampling_2x = \
            network_blocks.convolutional_block_3x3x3_stride_2(
            number_of_features, 2 * number_of_features)
        self._smoothing = network_blocks.convolutional_block_3x3x3(
            2 * number_of_features, 2 * number_of_features)

    def forward(self, block_input):
        output_of_downsampling_2x = self._downsampling_2x(block_input)
        return output_of_downsampling_2x, self._smoothing(
            output_of_downsampling_2x)


class ExpansionBlock3d(nn.Module):
    """Expansion block, that upsamples the input.

    The expansion blocks constitute the expansion part of
    the regularization network. Each block consists of 2x
    "upsampling" transposed convolution and
    conventional "smoothing" convolution. The output of the
    "upsampling" convolution is summed with the
    "shortcut_from_contraction" and is fed to the "smoothing"
    convolution.
    """

    def __init__(self, number_of_features):
        super(ExpansionBlock3d, self).__init__()
        self._upsampling_2x = \
            network_blocks.transposed_convolutional_block_4x4x4_stride_2(
                    number_of_features, number_of_features // 2)
        self._smoothing = network_blocks.convolutional_block_3x3x3(
            number_of_features // 2, number_of_features // 2)

    def forward(self, block_input, shortcut_from_contraction):
        output_of_upsampling = self._upsampling_2x(block_input)
        return self._smoothing(output_of_upsampling +
                               shortcut_from_contraction)


class LayernormConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, activation_function=None):
        super(LayernormConvLSTMCell, self).__init__()

        self.activation_function = activation_function

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2

        self.conv = nn.Conv3d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=False)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        b, c, d, h, w = h_cur.size()
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)

        cc_g = torch.layer_norm(cc_g, [d, h, w])
        g = self.activation_function(cc_g)

        c_next = f * c_cur + i * g
        c_next = torch.layer_norm(c_next, [d, h, w])
        h_next = o * self.activation_function(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        disparity, height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, disparity, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, disparity, height, width, device=self.conv.weight.device))


class LSTMFusion(torch.nn.Module):
    def __init__(self, input_dim):
        super(LSTMFusion, self).__init__()

        input_size = input_dim #  hyper_channels * 16
        hidden_size = input_dim # hyper_channels * 16

        self.lstm_cell = LayernormConvLSTMCell(input_dim=input_size,
                                               hidden_dim=hidden_size,
                                               kernel_size=(3, 3, 3),
                                               activation_function=torch.celu)

    def forward(self, current_encoding, current_state):
        batch, channel, disparity, height, width = current_encoding.size()

        if current_state is None:
            hidden_state, cell_state = self.lstm_cell.init_hidden(batch_size=batch,
                                                                  image_size=(disparity, height, width))
        else:
            hidden_state, cell_state = current_state

        next_hidden_state, next_cell_state = self.lstm_cell(input_tensor=current_encoding,
                                                            cur_state=[hidden_state, cell_state])

        return next_hidden_state, next_cell_state


class Regularization(nn.Module):
    """Regularization module, that enforce stereo matching constraints.

    It is a hourglass 3D convolutional network that consists
    of contraction and expansion parts, with the shortcut connections
    between them.

    The network downsamples the input 16x times along the spatial
    and disparity dimensions and then upsamples it 64x times along
    the spatial dimensions and 32x times along the disparity
    dimension, effectively computing matching cost only for even
    disparities.
    """

    def __init__(self, number_of_features=8, overall=True):
        """Returns initialized regularization module."""
        super(Regularization, self).__init__()
        self._smoothing = network_blocks.convolutional_block_3x3x3(
            number_of_features, number_of_features)
        self._contraction_blocks = nn.ModuleList([
            ContractionBlock3d(number_of_features * scale)
            for scale in [1, 2, 4]
        ])
        self._expansion_blocks = nn.ModuleList([
            ExpansionBlock3d(number_of_features * scale)
            for scale in [8, 4, 2]
        ])
        self._overall = overall
        if self._overall:
            self._upsample_to_halfsize = \
                network_blocks.transposed_convolutional_block_4x4x4_stride_2(
                    number_of_features, number_of_features // 2)
            self._upsample_to_fullsize = \
                network_blocks.transposed_convolution_3x4x4_stride_122(
                    number_of_features // 2, 1)

        self.lstm_fusion = LSTMFusion(input_dim=number_of_features * 8)

    def forward(self, matching_signatures, shortcut_from_left_image, lstm_state_bottom):
        """Returns regularized matching cost tensor.

        Args:
            matching_signatures: concatenated compact matching signatures
                                 for every disparity. It is tensor of size
                                 (batch_size, number_of_features,
                                 maximum_disparity / 4, height / 4,
                                 width / 4).
            shortcut_from_left_image: shortcut connection from the left
                                 image descriptor. It has size of
                                 (batch_size, number_of_features, height / 4,
                                  width / 4);

        Returns:
            regularized matching cost tensor of size (batch_size,
            maximum_disparity / 2, height, width). Every element of this
            tensor along the disparity dimension is a matching cost for
            disparity 0, 2, .. , maximum_disparity.
        """
        shortcuts_from_contraction = []
        shortcut = shortcut_from_left_image.unsqueeze(2)
        output = self._smoothing(matching_signatures)
        for contraction_block in self._contraction_blocks:
            shortcuts_from_contraction.append(output)
            shortcut, output = contraction_block(shortcut + output)

        bottom = output
        lstm_state_bottom = self.lstm_fusion(current_encoding=bottom,
                                             current_state=lstm_state_bottom)
        output = lstm_state_bottom[0]

        del shortcut
        for expansion_block in self._expansion_blocks:
            output = expansion_block(output, shortcuts_from_contraction.pop())

        if self._overall:
            return self._upsample_to_fullsize(
                self._upsample_to_halfsize(output)).squeeze_(1), lstm_state_bottom
        else:
            return output, lstm_state_bottom
