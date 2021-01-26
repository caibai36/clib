from typing import Union

import torch
from torch import nn

class CrossEntropyLossLabelSmoothing(nn.Module):
    """ Cross entropy loss function support label smoothing and weight of classes.

    https://arxiv.org/abs/1512.00567 Section 7. Model Regularization via Label Smoothing
    smoothed_loss = (1 - label_smoothing) * H(label_prob, model_prob) + label_smoothing * H(label_prob, uniform_prob)

    Cross entropy between two true distribution 'prob1' and model distribution 'prob2'
    (https://en.wikipedia.org/wiki/Cross_entropy)
    H(prob1, prob2) = -sum(prob1 *log(prob2))

    Paramters
    ---------
    label_smoothing: ratio smoothed by the uniform distribution
    weight: weight of the each type of class; shape (num_classes) (e.g., setting the weight zero when target is the padding label)
    reduction: 'mean' or 'sum' or 'none'; take the 'mean' or 'sum' of loss over batch, or return loss per batch if 'none'.

    Paramters of forward function
    -----------------------------
    source: shape (batch_size, num_classes) (or (batch_size * seq_length, num_classes))
    target: shape (batch_size) or (batch_size * seq_length)

    Returns of forward function
    ---------------------------
    loss: shape (batch_size) or (batch_size * seq_length) if reduction is 'none'
    or shape () if reduction is 'mean' or 'sum'

    Example
    -------
    Input:
    source = torch.Tensor([[0.9, 0.2, 0.3], [0.1, 0.9, 0.3], [0.9, 0.2, 0.3]])
    target = torch.LongTensor([1, 2, 1])

    label_smoothing = 0.8
    weight = torch.Tensor([0.1, 0.5, 0.4])
    reduction = 'none'

    ce_loss = CrossEntropyLossLabelSmoothing(label_smoothing=label_smoothing, weight=weight, reduction=reduction)
    print(ce_loss(source, target))

    Output:
    tensor([0.6011, 0.4742, 0.6011])
    """
    def __init__(self,
                 label_smoothing: float = 0.,
                 weight: Union[torch.Tensor, None] = None,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.reduction = reduction

        assert reduction == 'sum' or reduction == 'mean' or reduction == 'none', \
            "unknown return eduction '{}', reduction should be 'none', 'sum' or 'mean'".format(reduction)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_model_prob = torch.nn.functional.log_softmax(source, dim=-1) # batch_size x num_classes
        cross_entropy_label_and_model = -log_model_prob.gather(dim=1, index=target.unsqueeze(1)).squeeze(1) # [batch_size]; ce per batch

        if(self.label_smoothing > 0):
            num_classes = source.shape[-1]
            # sometimes '1/(num_classes-2)' to exclude <sos> and <eos>
            uniform_prob = torch.ones_like(source) * (1 / num_classes) # [batch_size * num_classes]
            cross_entropy_uniform_and_model = -(log_model_prob * uniform_prob).sum(dim=-1) # [batch_size]; cross entropy per batch
            cross_entropy_mixed = (1 - self.label_smoothing) * cross_entropy_label_and_model + \
                                  self.label_smoothing * cross_entropy_uniform_and_model
        else:
            cross_entropy_mixed = cross_entropy_label_and_model # shape of (batch_size)

        if self.weight is not None:
            cross_entropy_mixed = cross_entropy_mixed * self.weight.index_select(dim=0, index=target) # shape of (batch_size)

        if (self.reduction == 'none'):
            return cross_entropy_mixed
        elif (self.reduction == 'sum'):
            return cross_entropy_mixed.sum(dim=-1)
        else:
            return cross_entropy_mixed.mean(dim=-1)
