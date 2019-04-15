import warnings

import torch
from torch.nn.functional import _Reduction
from torch._C import _infer_size
from torch.nn.modules.loss import _WeightedLoss
from torch import nn
from torch.nn import functional as F
from torch.utils.data.sampler import Sampler


def binary_cross_entropy_class_weighted(input, target, weight=None, size_average=None,
                                        reduce=None, reduction='elementwise_mean',
                                        class_weight=None):
    r"""Function that measures the Binary Cross Entropy
    between the target and the output.

    See :class:`~torch.nn.BCELoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'
        class_weight: A list/tuple of two alpha values, summing up to 1, which indicate the relative weight
            of each class.

    Examples::

        >>> input = torch.randn((3, 2), requires_grad=True)
        >>> target = torch.rand((3, 2), requires_grad=False)
        >>> loss = F.binary_cross_entropy(F.sigmoid(input), target)
        >>> loss.backward()
    """

    # import numpy as np
    # print('max input:', np.max(input.cpu().data.numpy()))
    # print('min input:', np.min(input.cpu().data.numpy()))
    eps = 1e-12
    input = torch.clamp(input, min=eps, max=1 - eps)
    # print('max input:', np.max(input.cpu().data.numpy()))
    # print('min input:', np.min(input.cpu().data.numpy()))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                      "Please ensure they have the same size.".format(target.size(), input.size()))
    if input.nelement() != target.nelement():
        raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
                         "!= input nelement ({})".format(target.nelement(), input.nelement()))

    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)

    if class_weight is not None:
        loss = class_weight[1] * (target * torch.log(input)) + \
               class_weight[0] * ((1 - target) * torch.log(1 - input))

        # loss = (target * torch.log(input)) + \
        #        ((1 - target) * torch.log(1 - input))

        mean_loss = torch.neg(torch.mean(loss))
        # print('mean_loss:', mean_loss.cpu().data.numpy())
        return mean_loss

    mean_loss = torch._C._nn.binary_cross_entropy(input, target, weight, reduction)
    # print('mean_loss:', mean_loss.cpu().data.numpy())
    return mean_loss


class BCELossClassWeighted(_WeightedLoss):
    r"""Creates a criterion that measures the Binary Cross Entropy
    between the target and the output:

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size. If reduce is ``True``, then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `y` should be numbers
    between 0 and 1.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'
        class_weight: A list/tuple of two alpha values which indicate the relative weight of each class.
            The weights are enforced to sum up to 2. This is intuitively set so that the weights [1,1]
            correspond to the non-weighted BCE loss.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If `reduce` is False, then `(N, *)`, same shape as
          input.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean',
                 class_weight=None):
        super(BCELossClassWeighted, self).__init__(weight, size_average, reduce, reduction)

        if class_weight is not None:
            assert (class_weight[0] + class_weight[1] == 2), "The class_weights (alpha) should sum up to 2."
        self.class_weight = class_weight

    def forward(self, input, target):
        return binary_cross_entropy_class_weighted(input, target, weight=self.weight, reduction=self.reduction,
                                                   class_weight=self.class_weight)


def dice_torch(pred, target):
    """This definition generalize to real valued pred and target vector; i.e. it is an estiamted dice.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    # have to use contiguous since they may from a torch.view op
    # todo(amirabdi): I guess this will only calculate an estimate of the true dice per sample
    # And I ma right as this considers the entire batch as a single nd volume: sigma(2*A1*A2)/sigma(A1+A2)

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)

    return (2. * intersection) / (A_sum + B_sum)


def dice_torch_per_sample(pred, target):
    dice = 0
    batch_size = pred.size(0)
    for i in range(batch_size):
        iflat = pred[0].contiguous().view(-1)
        tflat = target[0].contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat)
        B_sum = torch.sum(tflat)
        dice += (2. * intersection) / (A_sum + B_sum)

    dice /= batch_size
    return dice


class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        diff = torch.abs(input - target)
        norm2 = torch.norm(diff, 2, dim=2)
        return torch.mean(norm2)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))


class DiceLossPerSample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1.

        # iflat = input.view(-1)
        # tflat = target.view(-1)
        # intersection = (iflat * tflat).sum()

        # return 1 - ((2. * intersection + smooth) /
        #            (iflat.sum() + tflat.sum() + smooth))
        dice = 0
        batch_size = input.size(0)
        for i in range(batch_size):
            iflat = input[0].contiguous().view(-1)
            tflat = target[0].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            A_sum = torch.sum(iflat)
            B_sum = torch.sum(tflat)
            dice += (2. * intersection + smooth) / (A_sum + B_sum + smooth)

        dice /= batch_size
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class SubsetIterativeSampler(Sampler):
    r"""Samples elements iteratively from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def mirror_torch(X, Y=None, dim_size=140, dim=0):
    '''
    :param X:
    :param Y:
    :param dim: The dimension (x=0, y=1, z=2) to flip the 3D data. This is not the same as axis in X
    :param dim_size: The number of voxels in dim
    :return:
    '''
    if len(X.size()) == 4:
        # first axis of X is the batch samples
        X = X.flip(dims=(dim + 1,))

        if Y is not None:
            if Y.size(1) == 28:
                LEFT_RIGHT_LANDMARK_MAPPING = LEFT_RIGHT_LANDMARK_MAPPING28
            else:
                raise NotImplementedError('Only 28 and 29 landmarks are supported for mirroring.')

            Y[:, :, dim] = dim_size - Y[:, :, dim]
            Y = Y[:, LEFT_RIGHT_LANDMARK_MAPPING, :]
    elif len(X.size()) == 3:
        # single sample to mirror
        X = X.flip(dims=(dim,))

        if Y is not None:
            if Y.size(0) == 28:
                LEFT_RIGHT_LANDMARK_MAPPING = LEFT_RIGHT_LANDMARK_MAPPING28
            else:
                raise NotImplementedError('Only 28 and 29 landmarks are supported for mirroring.')

            Y[:, dim] = dim_size - Y[:, dim]
            Y = Y[LEFT_RIGHT_LANDMARK_MAPPING, :]
    else:
        raise NotImplementedError

    return X if (Y is None) else (X, Y)