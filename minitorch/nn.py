from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    #raise NotImplementedError("Need to implement for Task 4.3"

     # Compute the new dimensions after pooling
    new_height = height // kh
    new_width = width // kw

    # Reshape the input to introduce kernel dimensions
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Rearrange dimensions to group the kernel elements together
    transposed = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Collapse the kernel dimensions into one
    tiled = transposed.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling on the input tensor.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width).
        kernel: Tuple representing the pooling kernel size (kh, kw).

    Returns:
    -------
        Pooled Tensor of shape (batch, channel, new_height, new_width).
    """
    # Tile the input tensor
    tiled, new_height, new_width = tile(input, kernel)

    # Compute the average over the tiled dimensions
    return tiled.mean(dim=4).view(input.shape[0], input.shape[1], new_height, new_width)


def max(input: Tensor, dim: int) -> Tuple[Tensor, Tensor]:
    """
    Compute the maximum value along a given dimension.

    Args:
    ----
        input: Input tensor.
        dim: The dimension to reduce.

    Returns:
    -------
        A tuple containing the maximum values and their indices.
    """
    max_values = input.f.add_reduce(input, dim)  # Replace with a compatible reduce operation
    max_indices = input.argmax(dim)  # Assuming argmax exists
    return max_values, max_indices


def softmax(input: Tensor, dim: int) -> Tensor:
    exp_values = input.exp()
    return exp_values / exp_values.sum(dim=dim)

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the log softmax along a given dimension.

    Args:
    ----
        input: Input tensor.
        dim: The dimension to apply log softmax.

    Returns:
    -------
        The log softmax tensor.
    """
    max_input = input.f.add_reduce(input, dim, keepdim=True)  # Replace with reduce
    log_sum_exp = (input - max_input).exp().sum(dim=dim, keepdim=True).log()
    return input - max_input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int], stride: Tuple[int, int] = None) -> Tensor:
    """
    Perform 2D max pooling on the input tensor.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width).
        kernel: Tuple representing the pooling kernel size (kh, kw).
        stride: Tuple representing the stride size (sh, sw). Defaults to the kernel size.

    Returns:
    -------
        Pooled Tensor of shape (batch, channel, new_height, new_width).
    """
    if stride is None:
        stride = kernel

    tiled, new_height, new_width = tile(input, kernel)
    return tiled.max(dim=4).view(input.shape[0], input.shape[1], new_height, new_width)

def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    Apply dropout to the input tensor.

    Args:
    ----
        input: Input tensor.
        p: Probability of dropping out units.
        training: Apply dropout if True.

    Returns:
    -------
        Tensor with dropout applied.
    """
    if not training:
        return input
    mask = (rand(input.shape) > p).float()
    return input * mask / (1 - p)