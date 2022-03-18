import torch
from torch import nn
import torch.nn.functional as F
from .loss_utils import class_weights


class CrossEntropyLoss(nn.Module):
    """
    Implements the cross entropy loss function.
    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'. Default ``'NCHW'``.
    """

    def __init__(self, weight=None, ignore_index=255, data_format='NCDHW'):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.EPS = 1e-8
        self.data_format = data_format
        if weight is not None:
            self.weight = torch.tensor(weight, dtype=torch.float)
        else:
            self.weight = None

    def forward(self, logit, label):
        """
        Forward computation.
        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        Returns:
            (Tensor): The average loss.
        """
        label = label.type(torch.int64)
        # label.shape: │[3, 128, 128, 128] logit.shape: [3, 3, 128, 128, 128]
        channel_axis = self.data_format.index("C")  # NCDHW -> 1, NDHWC -> 4

        if len(logit.shape) == 4:
            logit = logit.unsqueeze(0)

        if self.weight is None:
            self.weight = class_weights(logit)

        if self.weight is not None and logit.shape[channel_axis] != len(
                self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), logit.shape[channel_axis]))

        if channel_axis == 1:
            logit = torch.permute(logit, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC

        loss = F.cross_entropy(logit + self.EPS,
                               label,
                               reduction='mean',
                               ignore_index=self.ignore_index,
                               weight=self.weight)

        return loss
