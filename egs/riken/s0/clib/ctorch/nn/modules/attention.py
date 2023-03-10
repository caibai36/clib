from typing import List, Dict, Tuple, Union, Any

import torch
from torch import nn
from torch.nn import functional as F

from clib.ctorch.utils.module_util import get_act

class DotProductAttention(nn.Module):
    """  Attention by dot product.
    https://arxiv.org/abs/1508.04025 "Effective MNT" section 3.1 formula (8) (dot version)

    DotProductAttention is a module that takes in a dict with key of 'query' and 'context'
    (alternative key of 'mask' and 'need_expected_context'),
    and returns a output dict with key ('p_context' and 'expected_context').

    It takes 'query' (batch_size x query_size) and 'context' (batch_size x context_length x context_size),
    returns the proportion of attention ('p_context': batch_size x context_length) the query pays to different parts of context
    and the expected context vector ('expected_context': batch_size x context_size)
    by taking weighted average over the context by the proportion of attention.

    Example
    -------
    Input:
    query = torch.Tensor([[3, 4], [3, 5]])
    context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    mask = torch.ByteTensor([[1, 1],[1, 0]])
    input = {'query': query, 'context': context, 'mask': mask}

    attention = DotProductAttention()
    output = attention(input)

    Output:
    {'p_context': tensor([[0.7311, 0.2689], [0.9933, 0.0067]]),
    'expected_context': tensor([[3.2689, 3.7311], [3.0000, 4.9933]])}
    """

    def __init__(self,
                 context_size: int = -1,
                 query_size: int = -1,
                 normalize: bool = False) -> None:
        super().__init__()
        self.normalize = normalize
        self.att_vector_size = context_size

    def compute_expected_context(self, p_context: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """ compute the expected context by taking the weighted (p_context) average.

        p_context: batch_size x context_length
        context: batch_size x context_length x context_size
        expected_context: batch_size x context_size
        """
        return torch.bmm(p_context.unsqueeze(-2), context).squeeze(-2)

    def forward(self, input: Dict) -> Dict:
        query = input['query'] # batch_size x query_size
        context = input['context'] # batch_size x context_length x context_size
        assert query.shape[-1] == context.shape[-1], \
            "The query_size ({}) and context_size ({}) need to be same for the DotProductAttention.".format(
                query.shape[-1], context.shape[-1])
        mask = input.get('mask', None)
        need_expected_context = input.get('need_expected_context', True)

        # score = dot_product(context,query) formula (8) of "Effective MNT".
        score = torch.bmm(context, query.unsqueeze(-1)).squeeze(-1) # batch_size x context_length
        if self.normalize: score = score / math.sqrt(query_size)
        if mask is not None: score.masked_fill_(mask==0, -1e9)
        p_context = F.softmax(score, dim=-1)
        expected_context = self.compute_expected_context(p_context, context) if need_expected_context else None
        return {'p_context': p_context,
                'expected_context': expected_context}

class MLPAttention(nn.Module):
    """  Attention by multilayer perception (mlp).
    https://arxiv.org/abs/1508.04025 "Effective MNT" section 3.1 formula (8) (concat version)

    score = V*tanh(W[context,query])

    MLPAttention will concatenate the query and context and pass them through two-layer mlp to get the probability (attention) over context.
    It is a module that takes in a dict with key of 'query' and 'context' (alternative key of 'mask' and 'need_expected_context'),
    and returns a output dict with key ('p_context' and 'expected_context').

    It takes 'query' (batch_size x query_size) and 'context' (batch_size x context_length x context_size),
    returns the proportion of attention ('p_context': batch_size x context_length) the query pays to different parts of context
    and the expected context vector ('expected_context': batch_size x context_size)
    by taking weighted average over the context by the proportion of attention.

    Example
    -------
    Input:
    query = torch.Tensor([[3, 4], [3, 5]])
    context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    mask = torch.ByteTensor([[1, 1],[1, 0]])
    input = {'query': query, 'context': context, 'mask': mask}

    attention = MLPAttention(context.shape[-1], query.shape[-1])
    output = attention(input)

    Output:
    {'p_context': tensor([[0.4997, 0.5003], [0.4951, 0.5049]], grad_fn=<SoftmaxBackward>),
    'expected_context': tensor([[3.5003, 3.4997], [3.0000, 4.4951]], grad_fn=<SqueezeBackward1>)}
    """

    def __init__(self,
                 context_size: int,
                 query_size: int,
                 att_hidden_size: int = 256,
                 att_act: str = 'tanh',
                 normalize: bool = True) -> None:
        super().__init__()
        self.concat2proj = nn.Linear(query_size+context_size, att_hidden_size) # W in formula (8) of "Effective MNT"
        self.att_act = get_act(att_act)()
        self.proj2score = nn.utils.weight_norm(nn.Linear(att_hidden_size, 1)) if normalize \
                          else nn.Linear(att_hidden_size, 1) # V in formula (8) of "Effective MNT"
        self.att_vector_size = context_size

    def compute_expected_context(self, p_context: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """ compute the expected context by taking the weighted (p_context) average.

        p_context: batch_size x context_length
        context: batch_size x context_length x context_size
        expected_contex: batch_size x context_size
        """
        return torch.bmm(p_context.unsqueeze(-2), context).squeeze(-2)

    def forward(self, input: Dict) -> Dict:
        query = input['query'] # batch_size x query_size
        batch_size, query_size = query.shape
        context = input['context'] # batch_size x context_length x context_size
        batch_size, context_length, context_size = context.shape
        mask = input.get('mask', None)
        need_expected_context = input.get('need_expected_context', True)

        # score = V*tanh(W[context,query]) formula (8) of "Effective MNT".
        concat = torch.cat([context, query.unsqueeze(-2).expand(batch_size, context_length, query_size)], dim=-1) # batch_size x context_length x (context_size + query_size)
        score = self.proj2score(self.att_act(self.concat2proj(concat))).squeeze(-1) # batch_size x context_length

        if mask is not None: score.masked_fill_(mask==0, -1e9)
        p_context = F.softmax(score, dim=-1)
        expected_context = self.compute_expected_context(p_context, context) if need_expected_context else None # batch_size x context_size
        return {'p_context': p_context,
                'expected_context': expected_context}
