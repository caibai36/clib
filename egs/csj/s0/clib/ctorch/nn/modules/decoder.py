from typing import List, Dict, Tuple, Union, Any

import torch
from torch import nn
from torch.nn import functional as F

from clib.ctorch.utils.module_util import get_rnn, get_act, get_att

class LuongDecoder(nn.Module):
    """ Implementation of the decoder of "Effective NMT" by Luong.
    https://arxiv.org/abs/1508.04025 "Effective MNT"
    section 3 formula (5) to create attentional vector
    section 3.3 the input feeding approach

    At time step t of decoder, the message flows as follows
    [attentional_vector[t-1], input] -> hidden[t] ->
    [expected_context_vector[t], hidden[t]] -> attentional_vector[t]

    Input feeding: concatenate the input with the attentional vector from
    last time step to combine the alignment information in the past.

    attentional vector: we get hidden state at the top the stacked LSTM layers,
    then concatenate the hidden state and expected context vector for linearly
    projecting (context_proj_*) to the attentional vector.
    see "Effective NMT" section 3 formula (5)
        attentional_vector = tanh(W[context_vector, hidden])

    Example
    -------
    In:
    input_embedding = torch.Tensor([[0.3, 0.4], [0.3, 0.5]])
    context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    context_mask = torch.ByteTensor([[1, 1],[1, 0]])

    luong_decoder = LuongDecoder(att_config={"type": "mlp"},
                                 context_size=context.shape[-1],
                                 input_size=input_embedding.shape[-1],
                                 rnn_sizes=[512, 512],
                                 rnn_config={"type": "lstmcell"},
                                 rnn_dropout=0.25,
                                 context_proj_size=3, # the size of attentional vector
                                 context_proj_act='tanh')

    luong_decoder.set_context(context, context_mask)
    output, att_out = luong_decoder(input_embedding)
    # output, att_out = luong_decoder(input_embedding, dec_mask=torch.Tensor([0, 1])) # mask the first instance in two batches
    print("output of Luong decoder: {}".format(output))
    print("output of attention layer: {}".format(att_out))

    Out:
    output of Luong decoder: tensor([[0.0268, 0.0782, 0.0374], [0.0285, 0.1341, 0.0169]], grad_fn=<TanhBackward>)
    output of attention layer: {'p_context': tensor([[0.4982, 0.5018], [0.4990, 0.5010]], grad_fn=<SoftmaxBackward>),
                         'expected_context': tensor([[3.5018, 3.4982], [3.0000, 4.4990]], grad_fn=<SqueezeBackward1>)}
    """

    def __init__(self,
                 att_config: Dict, # configuration of attention
                 context_size: int,
                 input_size: int,
                 rnn_sizes: List[int] = [512, 512],
                 rnn_config: Dict = {"type": "lstmcell"},
                 rnn_dropout: Union[List[float], float] = 0.25,
                 context_proj_size: int = 256, # the size of attentional vector
                 context_proj_act: str = 'tanh',
                 context_proj_dropout: int = 0.25) -> None:
        super().__init__()

        # Copy the configuration for each layer
        num_rnn_layers = len(rnn_sizes)
        if not isinstance(rnn_dropout, list): rnn_dropout = [rnn_dropout] * num_rnn_layers # sometimes dropout not at the top layer
        assert num_rnn_layers == len(rnn_dropout), "The number of rnn layers does not match length of rnn_dropout list."

        self.att_config = att_config
        self.context_size = context_size
        self.input_size = input_size
        self.rnn_sizes = rnn_sizes
        self.rnn_config = rnn_config
        self.rnn_dropout = rnn_dropout
        self.context_proj_size = context_proj_size
        self.context_proj_act = context_proj_act
        self.context_proj_dropout = context_proj_dropout

        self.num_rnn_layers = num_rnn_layers

        # Initialize attentional vector of previous time step
        self.attentional_vector_pre = None

        # Initialize stacked rnn layers with their hidden states and cell states
        self.rnn_layers = nn.ModuleList()
        self.rnn_hidden_cell_states = []
        pre_size = input_size + context_proj_size # input feeding
        for i in range(num_rnn_layers):
            self.rnn_layers.append(get_rnn(rnn_config['type'])(pre_size, rnn_sizes[i]))
            self.rnn_hidden_cell_states.append(None) # initialize (hidden state, cell state) of each layer as Nones.
            pre_size = rnn_sizes[i]

        # Get expected context vector from attention
        self.attention_layer = get_att(att_config['type'])(context_size, pre_size)

        # Combine hidden state and context vector to be attentional vector.
        self.context_proj_layer = nn.Linear(pre_size + context_size, context_proj_size)

        self.output_size = context_proj_size

    def set_context(self, context: torch.Tensor, context_mask: Union[torch.Tensor, None] = None) -> None:
        self.context = context
        self.context_mask = context_mask

    def get_context_and_its_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.context, self.context_mask

    def reset(self) -> None:
        """ Reset the the luong decoder
        by setting the attentional vector of the previous time step to be None,
        which means forgetting all the formation
        accumulated in the history (by RNN) before the current time step
        and forgetting the attention information of the previous time step.
        """
        self.attentional_vector_pre = None
        for i in range(self.num_rnn_layers):
            self.rnn_hidden_cell_states[i] = None

    def decode(self, input: torch.Tensor, dec_mask: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, Dict]:
        """
        input: batch_size x input_size
        dec_mask: batch_size
        # target batch 3 with length 2, 1, 3 => mask = [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
        # Each time step corresponds to each column of the mask.
        # In time step 2, the second column [1, 0, 1] as the dec_mask
        # dec_mask with shape [batch_size]
        # target * dec_mask.unsqueeze(-1).expand_as(target) will mask out
        # the feature of the second element of batch at time step 2, while the element with length 1
        """
        batch_size, input_size = input.shape
        if self.attentional_vector_pre is None:
            self.attentional_vector_pre = input.new_zeros(batch_size, self.context_proj_size)

        # Input feeding: initialize the input of LSTM with previous attentional vector information
        output = torch.cat([input, self.attentional_vector_pre], dim=-1)
        for i in range(self.num_rnn_layers):
            output, cell = self.rnn_layers[i](output, self.rnn_hidden_cell_states[i]) # LSTM cell return (h, c)
            self.rnn_hidden_cell_states[i] = (output, cell) # store the hidden state and cell state of current layer for next time step.
            if dec_mask is not None: output = output * dec_mask.unsqueeze(-1).expand_as(output)

            output = F.dropout(output, p=self.rnn_dropout[i], training=self.training)

        # Get the context vector from the hidden state at the top of rnn layers.
        att_out = self.attention_layer({'query': output, 'context': self.context, 'mask': self.context_mask})

        # Get the attentional vector of current time step by linearly projection from hidden state and context vector
        act_func = get_act(self.context_proj_act)()
        output = act_func(self.context_proj_layer(torch.cat([output, att_out['expected_context']], dim = -1)))

        if dec_mask is not None: output = output * dec_mask.unsqueeze(-1).expand_as(output)

        self.attentional_vector_pre = output # attentional vector before dropout might be more stable

        output = F.dropout(output, p=self.context_proj_dropout, training=self.training)

        return output, att_out
