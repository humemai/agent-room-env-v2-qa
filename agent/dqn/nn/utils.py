"""A lot copied from https://github.com/migalkin/StarE"""

import torch
import torch_scatter
from torch_scatter import scatter_add, scatter_max


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def softmax(src, index, num_nodes=None) -> torch.Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    Returns:
        out: The softmax values.

    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return torch.irfft(
        com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],)
    )


def ccorr(a, b):
    return torch.irfft(
        com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)),
        1,
        signal_sizes=(a.shape[-1],),
    )


def rotate(h, r):
    # re: first half, im: second half
    # assume embedding dim is the last dimension
    d = h.shape[-1]
    h_re, h_im = torch.split(h, d // 2, -1)
    r_re, r_im = torch.split(r, d // 2, -1)
    return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)


def scatter_(name, src, index, dim_size=None) -> torch.Tensor:
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    Returns:
        Tensor: The aggregated tensor.
    """

    assert name in ["add", "mean", "max"]

    op = getattr(torch_scatter, "scatter_{}".format(name))
    fill_value = -1e38 if name == "max" else 0
    out = op(src, index, 0, None, dim_size, fill_value)
    if isinstance(out, tuple):
        out = out[0]

    if name == "max":
        out[out == fill_value] = 0

    return out
