from typing import List, Dict, Tuple, Union, Any

import torch
from torch import nn
from torch.nn import functional as F

from clib.ctorch.utils.tensor_util import mask2length
from clib.ctorch.nn.search.utils import crop_hypothesis_lengths

def greedy_search_torch(model: nn.Module,
                       source: torch.Tensor,
                       source_lengths: torch.Tensor,
                       sos_id: int,
                       eos_id: int,
                       max_dec_length: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor]:
    """ Generate the hypothesis from source by greedy search (beam search with beam_size 1)

    Parameters
    ----------
    model: an attention sequence2sequence model
    source: shape of [batch_size, source_max_length, source_size]
    source_length: shape of [batch_size]
    sos_id: id of the start of sequence token
    eos_id: id of the end of sequence token
    max_dec_length: the maximum length of the hypothesis (a sequence of tokens)
        the decoder can generate, even if eos token does not occur.

    Returns
    -------
    hypothesis: shape [batch_size, dec_length]; each hypothesis is a sequence of tokenid
        (which has no sos_id, but with eos_id if its length is less than max_dec_length)
    lengths of hypothesis: shape [batch_size]; length without sos_id but with eos_id
    attentions of hypothesis: shape [batch_size, dec_length, context_size]
    presoftmax of hypothesis: shape [batch_size, dec_length, dec_output_size]
    """
    model.reset()
    model.train(False)
    model.encode(source, source_lengths) # set the context for decoding at the same time

    batch_size = source.shape[0]

    hypo_list = [] # list of different time steps
    hypo_att_list = []
    hypo_presoftmax = []
    hypo_lengths = source.new_full([batch_size], -1).long()
    cur_tokenids = source.new_full([batch_size], sos_id).long()
    for time_step in range(max_dec_length):
        presoftmax, dec_att = model.decode(cur_tokenids)
        next_tokenids = presoftmax.argmax(-1) # [batch_size]
        hypo_list.append(next_tokenids)
        hypo_att_list.append(dec_att['p_context'])
        hypo_presoftmax.append(presoftmax)

        for i in range(batch_size):
            if next_tokenids[i] == eos_id and hypo_lengths[i] == -1:
                hypo_lengths[i] = time_step + 1
        if all(hypo_lengths != -1): break
        cur_tokenids = next_tokenids

    hypo = torch.stack(hypo_list, dim=1) # [batch_size, dec_length]
    hypo_att = torch.stack(hypo_att_list, dim=1) # [batch_size, dec_length, context_size]
    hypo_presoftmax = torch.stack(hypo_presoftmax, dim=1) # [batch_size, dec_length, dec_output_size]
    return hypo, hypo_lengths, hypo_att, hypo_presoftmax

def greedy_search(model: nn.Module,
                  source: torch.Tensor,
                  source_lengths: torch.Tensor,
                  sos_id: int,
                  eos_id: int,
                  max_dec_length: int) -> Tuple[List[torch.LongTensor], torch.LongTensor, List[torch.Tensor]]:
    """ Generate the hypothesis from source by greedy search (beam search with beam_size 1)

    Parameters
    ----------
    model: an attention sequence2sequence model
    source: shape of [batch_size, source_max_length, source_size]
    source_length: shape of [batch_size]
    sos_id: id of the start of sequence token
    eos_id: id of the end of sequence token
    max_dec_length: the maximum length of the hypothesis (a sequence of tokens)
        the decoder can generate, even if eos token does not occur.

    Returns
    -------
    cropped hypothesis: a list of [hypo_lengths[i]] tensors with the length batch_size.
        each element in the batch is a sequence of tokenids excluding eos_id.
    cropped lengths of hypothesis: shape [batch_size]; excluding sos_id and eos_id
    cropped attentions of hypothesis: a list of [hypo_lengths[i], context_length[i]] tensors
        with the length batch_size
    cropped presoftmax of hypothesis: a list of [hypo_lengths[i], dec_output_size] tensors

    Example
    -------
    Input:
    token2id, first_batch, model = get_token2id_firstbatch_model()
    source, source_lengths = first_batch['feat'], first_batch['num_frames']
    sos_id, eos_id = int(token2id['<sos>']), int(token2id['<eos>']) # 2, 3
    max_dec_length = 5

    model.to(device)
    source, source_lengths = source.to(device), source_lengths.to(device)
    hypo, hypo_lengths, hypo_att, hypo_presoftmax = greedy_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length)
    cropped_hypo, cropped_hypo_lengths, cropped_hypo_att, cropped_hypo_presoftmax = greedy_search(model, source, source_lengths, sos_id, eos_id, max_dec_length)

    Output:
    ---hypo---
    tensor([[5, 4, 3, 3],
            [5, 4, 5, 4],
            [5, 6, 3, 3]])
    ---hypo_lengths---
    tensor([ 3, -1,  3])
    ---hypo_att---
    tensor([[[0.0187, 0.9813],
             [0.0210, 0.9790],
             [0.0193, 0.9807],
             [0.0201, 0.9799]],

            [[0.0057, 0.9943],
             [0.0056, 0.9944],
             [0.0050, 0.9950],
             [0.0056, 0.9944]],

            [[1.0000, 0.0000],
             [1.0000, 0.0000],
             [1.0000, 0.0000],
             [1.0000, 0.0000]]], grad_fn=<StackBackward>)
    ---hypo_presoftmax---
    tensor([[[-2.0391e+00, -2.4686e+00, -3.3696e+00, -1.2013e+00, -1.9056e+00, 4.6328e+00,  1.2401e-01, -9.1314e-01],
             [-4.5584e+00, -5.5560e+00, -1.8747e+00,  2.1034e-03,  1.0197e+00, -3.7233e+00,  9.6586e-01,  2.8960e-02],
             [-3.4743e+00, -4.5990e+00, -2.4292e+00,  6.7183e-01, -6.9239e-02, -2.2819e+00,  5.3374e-01,  9.2140e-03],
             [-3.8338e+00, -5.1679e+00, -1.9896e+00,  8.6487e-01,  5.4353e-01, -3.8287e+00,  5.9950e-01,  2.5497e-01]],

            [[-1.0977e+00, -1.8111e+00, -3.2346e+00, -9.9084e-01, -2.3206e+00, 5.5821e+00, -3.4452e-01, -7.9397e-01],
             [-3.1162e+00, -4.4986e+00, -1.2099e+00, -6.0075e-02,  6.6851e-01, -2.0799e+00,  2.1094e-01,  2.7038e-01],
             [-1.5080e+00, -2.7002e+00, -2.3081e+00, -2.9946e-01, -1.3555e+00, 2.6545e+00, -4.2277e-01, -1.3397e-01],
             [-3.0643e+00, -4.4616e+00, -1.1970e+00, -2.8974e-02,  6.4926e-01, -2.0641e+00,  1.8507e-01,  2.8324e-01]],

            [[-2.2006e+00, -2.2896e+00, -3.6796e+00, -1.0538e+00, -1.8577e+00, 4.2987e+00,  5.3117e-01, -1.2819e+00],
             [-4.5086e+00, -4.8001e+00, -2.4802e+00, -1.3172e-01,  9.3378e-01, -3.6198e+00,  1.4054e+00, -6.8509e-01],
             [-2.6262e+00, -3.4670e+00, -2.7019e+00,  1.9906e+00, -3.1856e-01, -3.5389e+00,  6.1016e-01, -2.3925e-01],
             [-3.5176e+00, -4.2128e+00, -2.5136e+00,  1.0150e+00,  3.2375e-01, -3.6914e+00,  9.3068e-01, -3.6401e-01]]], grad_fn=<StackBackward>)

    ---cropped_hypo---
    [tensor([5, 4]), tensor([5, 4, 5, 4]), tensor([5, 6])]
    ---cropped_hypo_lengths---
    tensor([2, 4, 2])
    ---cropped_hypo_att---
    [tensor([[0.0187, 0.9813],
            [0.0210, 0.9790]], grad_fn=<SliceBackward>),
     tensor([[0.0057, 0.9943],
            [0.0056, 0.9944],
            [0.0050, 0.9950],
            [0.0056, 0.9944]], grad_fn=<AliasBackward>),
     tensor([[1.],
            [1.]], grad_fn=<SliceBackward>)]
    ---cropped_hypo_presoftmax---
    [tensor([[-2.0391e+00, -2.4686e+00, -3.3696e+00, -1.2013e+00, -1.9056e+00, 4.6328e+00,  1.2401e-01, -9.1314e-01],
             [-4.5584e+00, -5.5560e+00, -1.8747e+00,  2.1034e-03,  1.0197e+00, -3.7233e+00,  9.6586e-01,  2.8960e-02]], grad_fn=<SliceBackward>),
     tensor([[-1.0977, -1.8111, -3.2346, -0.9908, -2.3206,  5.5821, -0.3445, -0.7940],
             [-3.1162, -4.4986, -1.2099, -0.0601,  0.6685, -2.0799,  0.2109,  0.2704],
             [-1.5080, -2.7002, -2.3081, -0.2995, -1.3555,  2.6545, -0.4228, -0.1340],
             [-3.0643, -4.4616, -1.1970, -0.0290,  0.6493, -2.0641,  0.1851,  0.2832]], grad_fn=<SliceBackward>),
     tensor([[-2.2006, -2.2896, -3.6796, -1.0538, -1.8577,  4.2987,  0.5312, -1.2819],
             [-4.5086, -4.8001, -2.4802, -0.1317,  0.9338, -3.6198,  1.4054, -0.6851]], grad_fn=<SliceBackward>)]
    """
    batch_size = source.shape[0]
    # shape: [batch_size, dec_length], [batch_size], [batch_size, dec_length, context_size]
    hypo, hypo_lengths, hypo_att, hypo_presoftmax = greedy_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length)

    context_lengths = mask2length(model.decoder.context_mask)
    cropped_hypo_lengths = crop_hypothesis_lengths(hypo_lengths, max_dec_length) # remove eos_id
    cropped_hypo = [hypo[i][0:cropped_hypo_lengths[i]] for i in range(batch_size)]
    cropped_hypo_att = [hypo_att[i][0:cropped_hypo_lengths[i], 0:context_lengths[i]] for i in range(batch_size)]
    cropped_hypo_presoftmax = [hypo_presoftmax[i][0:cropped_hypo_lengths[i], :] for i in range(batch_size)]

    return cropped_hypo, cropped_hypo_lengths, cropped_hypo_att, cropped_hypo_presoftmax
