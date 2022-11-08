from typing import Union
import torch

def length2mask(sequence_lengths: torch.Tensor, max_length: Union[int, None] = None) -> torch.Tensor:
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.

    Examples
    --------
    sequence_lengths: [2, 2, 3], max_length: 4: -> mask: [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]

    In [451]: lengths = torch.tensor([2, 2, 3])
    In [452]: length2mask(lengths, 4)
    Out[452]:
    tensor([[1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0]])

    In [453]: length2mask(lengths, 2)
    Out[453]:
    tensor([[1, 1],
            [1, 1],
            [1, 1]])

    In [276]: length2mask(lengths)
    Out[276]:
    tensor([[1, 1, 0],
            [1, 1, 0],
            [1, 1, 1]])
    """
    if max_length is None:
        max_length = int(torch.max(sequence_lengths).item())
    ones_seqs = sequence_lengths.new_ones(len(sequence_lengths), max_length)
    cumsum_ones = ones_seqs.cumsum(dim=-1)

    return (cumsum_ones <= sequence_lengths.unsqueeze(-1)).long()

def mask2length(mask: torch.Tensor) -> torch.LongTensor:
    """
    Compute sequence lengths for the batch from a binary mask.

    Parameters
    ----------
    mask: a binary mask of shape [batch_size, sequence_length]

    Returns
    -------
    the lengths of the sequences in the batch of shape [batch_size]

    Example
    -------
    In [458]: mask
    Out[458]:
    tensor([[1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0]])

    In [459]: mask2length(mask)
    Out[459]: tensor([2, 2, 3])
    """
    return mask.long().sum(-1)
