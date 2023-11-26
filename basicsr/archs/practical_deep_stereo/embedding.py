# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

from torch import nn

from . import network_blocks


class Embedding(nn.Module):
    """Embedding module."""

    def __init__(self,
                 number_of_input_features=3,
                 number_of_embedding_features=64,
                 number_of_shortcut_features=8,
                 number_of_residual_blocks=2, shortcut=True):
        """Returns initialized embedding module.

        Args:
            number_of_input_features: number of channels in the input image;
            number_of_embedding_features: number of channels in image's
                                          descriptor;
            number_of_shortcut_features: number of channels in the redirect
                                         connection descriptor;
            number_of_residual_blocks: number of residual blocks in embedding
                                       network.
        """
        super(Embedding, self).__init__()
        embedding_modules = [
            nn.InstanceNorm2d(number_of_input_features),
            network_blocks.convolutional_block_5x5_stride_2(
                number_of_input_features, number_of_embedding_features),
            network_blocks.convolutional_block_5x5_stride_2(
                number_of_embedding_features, number_of_embedding_features),
        ]
        embedding_modules += [
            network_blocks.ResidualBlock(number_of_embedding_features)
            for _ in range(number_of_residual_blocks)
        ]
        self._embedding_modules = nn.ModuleList(embedding_modules)
        self.shortcut = shortcut
        if self.shortcut:
            self._shortcut = network_blocks.convolutional_block_3x3(
                number_of_embedding_features, number_of_shortcut_features)

    def forward(self, image):
        """Returns image's descriptor and redirect connection descriptor.

        Args:
            image: color image of size
                   batch_size x 3 x height x width;

        Returns:
            descriptor: image's descriptor of size
                        batch_size x 64 x (height / 4) x (width / 4);
            shortcut_from_left_image: shortcut connection from left image
                      descriptor (it is used in regularization network). It
                      is tensor of size
                      (batch_size, 8, height / 4, width / 4).
        """
        descriptor = image
        for embedding_module in self._embedding_modules:
            descriptor = embedding_module(descriptor)
        if self.shortcut:
            return descriptor, self._shortcut(descriptor)
        else:
            return descriptor
