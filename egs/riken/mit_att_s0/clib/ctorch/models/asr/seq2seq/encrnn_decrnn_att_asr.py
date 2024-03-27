from typing import List, Dict, Tuple, Union, Any
import torch
from torch import nn
from torch.nn import functional as F

from clib.ctorch.nn.modules.encoder import PyramidRNNEncoder
from clib.ctorch.nn.modules.decoder import LuongDecoder

class EncRNNDecRNNAtt(nn.Module):
    """ Sequence-to-sequence module with RNN encoder and RNN decoder with attention mechanism.

    The default encoder is pyramid RNN Encoder (https://arxiv.org/abs/1508.01211 "LAS" section 3.1 formula (5)),
    which passes the input feature through forward feedback neural network ('enc_fnn_*') and RNN ('enc_rnn_*').
    Between the RNN layers we use the subsampling ('enc_rnn_subsampling_*')
by concatenating both frames or taking the first frame every two frames.

    The default decoder is Luong Decoder (https://arxiv.org/abs/1508.04025 "Effective MNT"
    section 3 formula (5) to create attentional vector; section 3.3 the input feeding approach)
    which passes the embedding of the input ('dec_embedding_*')
    along with previous attentional vector into a stacked RNN layers ('dec_rnn_*'),
    linearly projects the top RNN hidden state and expected context vector to the current attentional vector ('dec_context_proj_*'),
    and feed the attentional vector to next time step.

    The default attention is the multilayer perception attention by concatenation
    (https://arxiv.org/abs/1508.04025 "Effective MNT" section 3.1 formula (8) (concat version))
    which concatenates the hidden state from decoder and each part of context from encoder
    and passes them to two layers neural network to get the alignment score.
    The scores are then normalized to get the probability of different part of context and get the expected context vector.

    We tie the embedding weight by default
    (https://arxiv.org/abs/1608.05859 "Using the Output Embedding to Improve Language Models" introduction),
    which shares the weights between (dec_onehot => dec_embedding) and (output_embedding(attentional vector) => pre_softmax)

    Information flow:
    encode:
    enc_input->encoder->context

    decode with one time step:
    dec_input->dec_embedding->dec_hidden
    (context,dec_hidden)->context_vector
    (context_vector,dec_hidden)->attentional_vector
    attentional_vector->pre_softmax
    """
    def __init__(self,
                 enc_input_size: int,
                 dec_input_size: int,
                 dec_output_size: int,
                 enc_fnn_sizes: List[int] = [512],
                 enc_fnn_act: str = 'relu',
                 enc_fnn_dropout: Union[float, List[float]] = 0.25,
                 enc_rnn_sizes: List[int] = [256, 256, 256],
                 enc_rnn_config: Dict = {'type': 'lstm', 'bi': True},
                 enc_rnn_dropout: Union[float, List[float]] = 0.25,
                 enc_rnn_subsampling: Union[bool, List[bool]] = [False, True, True],
                 enc_rnn_subsampling_type: str = 'concat', # 'concat' or 'drop'
                 dec_embedding_size: int = 256,
                 dec_embedding_dropout: float = 0.25,
                 # share weights between (input_onehot => input_embedding) and (output_embedding => pre_softmax)
                 dec_embedding_weights_tied: bool = True,
                 dec_rnn_sizes: List[int] = [512, 512],
                 dec_rnn_config: Dict = {"type": "lstmcell"},
                 dec_rnn_dropout: Union[List[float], float] = 0.25,
                 dec_context_proj_size: int = 256, # the size of attentional vector
                 dec_context_proj_act: str = 'tanh',
                 dec_context_proj_dropout: int = 0.25,
                 enc_config: Dict = {'type': 'pyramid_rnn_encoder'},
                 dec_config: Dict = {'type': 'luong_decoder'},
                 att_config: Dict = {'type': 'mlp'}, # configuration of attention
    ) -> None:
        super().__init__()

        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.dec_output_size = dec_output_size
        self.enc_fnn_sizes = enc_fnn_sizes
        self.enc_fnn_act = enc_fnn_act
        self.enc_fnn_dropout = enc_fnn_dropout
        self.enc_rnn_sizes = enc_rnn_sizes
        self.enc_rnn_config = enc_rnn_config
        self.enc_rnn_dropout = enc_rnn_dropout
        self.enc_rnn_subsampling = enc_rnn_subsampling
        self.enc_rnn_subsampling_type = enc_rnn_subsampling_type
        self.dec_embedding_size = dec_embedding_size
        self.dec_embedding_dropout = dec_embedding_dropout
        self.dec_embedding_weights_tied = dec_embedding_weights_tied
        self.dec_rnn_sizes = dec_rnn_sizes
        self.dec_rnn_config = dec_rnn_config
        self.dec_rnn_dropout = dec_rnn_dropout
        self.dec_context_proj_size = dec_context_proj_size
        self.dec_context_proj_act = dec_context_proj_act
        self.dec_context_proj_dropout = dec_context_proj_dropout
        self.enc_config = enc_config
        self.dec_config = dec_config
        self.att_config = att_config

        assert enc_config['type'] == 'pyramid_rnn_encoder', \
            "The encoder type '{}' is not implemented. Supported types include 'pyramid_rnn_encoder'.".format(enc_config['type'])
        assert dec_config['type'] == 'luong_decoder', \
            "The decoder type '{}' is not implemented. Supported types include 'luong_encoder'.".format(dec_config['type'])

        # Encoder
        self.encoder = PyramidRNNEncoder(enc_input_size=enc_input_size,
                                         enc_fnn_sizes=enc_fnn_sizes,
                                         enc_fnn_act=enc_fnn_act,
                                         enc_fnn_dropout=enc_fnn_dropout,
                                         enc_rnn_sizes=enc_rnn_sizes,
                                         enc_rnn_config=enc_rnn_config,
                                         enc_rnn_dropout=enc_rnn_dropout,
                                         enc_rnn_subsampling=enc_rnn_subsampling,
                                         enc_rnn_subsampling_type=enc_rnn_subsampling_type,
                                         enc_input_padding_value=0.0)
        self.enc_context_size = self.encoder.get_context_size()

        # Embedder
        self.dec_embedder = nn.Embedding(dec_input_size, dec_embedding_size, padding_idx=None)

        # Decoder
        self.decoder = LuongDecoder(att_config=att_config,
                                    context_size=self.enc_context_size,
                                    input_size=dec_embedding_size,
                                    rnn_sizes=dec_rnn_sizes,
                                    rnn_config=dec_rnn_config,
                                    rnn_dropout=dec_rnn_dropout,
                                    context_proj_size=dec_context_proj_size,
                                    context_proj_act=dec_context_proj_act,
                                    context_proj_dropout=dec_context_proj_dropout)

        # Presoftmax
        self.dec_presoftmax = nn.Linear(dec_context_proj_size, dec_output_size) # decoder.output_size == dec_context_proj_size

        # Typing weight
        if (dec_embedding_weights_tied):
            assert (dec_input_size, dec_embedding_size) == (dec_output_size, dec_context_proj_size), \
                f"When tying embedding weights: the shape of embedder weights: " + \
                f"(dec_input_size = {dec_input_size}, dec_embedding_size = {dec_embedding_size})\n" + \
                f"should be same as the shape of presoftmax weights: " + \
                f"(dec_output_size = {dec_output_size}, dec_context_proj_size = {dec_context_proj_size})"
            # tie weights between dec_embedder(input_onehot => input_embedding) and presoftmax(output_embedding => pre_softmax)
            self.dec_presoftmax.weight = self.dec_embedder.weight

    def get_config(self) -> Dict:
        return {'class': str(self.__class__),
                'enc_input_size': self.enc_input_size,
                'dec_input_size': self.dec_input_size,
                'dec_output_size': self.dec_output_size,
                'enc_fnn_sizes': self.enc_fnn_sizes,
                'enc_fnn_act': self.enc_fnn_act,
                'enc_fnn_dropout': self.enc_fnn_dropout,
                'enc_rnn_sizes': self.enc_rnn_sizes,
                'enc_rnn_config': self.enc_rnn_config,
                'enc_rnn_dropout': self.enc_rnn_dropout,
                'enc_rnn_subsampling': self.enc_rnn_subsampling,
                'enc_rnn_subsampling_type': self.enc_rnn_subsampling_type,
                'dec_embedding_size': self.dec_embedding_size,
                'dec_embedding_dropout': self.dec_embedding_dropout,
                'dec_embedding_weights_tied': self.dec_embedding_weights_tied,
                'dec_rnn_sizes': self.dec_rnn_sizes,
                'dec_rnn_config': self.dec_rnn_config,
                'dec_rnn_dropout': self.dec_rnn_dropout,
                'dec_context_proj_size': self.dec_context_proj_size,
                'dec_context_proj_act': self.dec_context_proj_act,
                'dec_context_proj_dropout': self.dec_context_proj_dropout,
                'enc_config': self.enc_config,
                'dec_config': self.dec_config,
                'att_config': self.att_config}

    def encode(self,
               enc_input: torch.Tensor,
               enc_input_lengths: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Paramters
        ---------
        enc_input: input feature with shape (batch_size x max_seq_length x in_size),
        enc_input_lengths: lengths of input with shape (batch_size) or None

        Returns
        -------
        the context vector (batch_size x max_context_length x context_size)
        the mask of context vector (batch_size x max_context_length).

        Note:
        We set the context for decoder when calling the encode function.
        """
        context, context_mask = self.encoder.encode(enc_input, enc_input_lengths)
        self.decoder.set_context(context, context_mask)
        return context, context_mask

    def decode(self,
               dec_input: torch.Tensor,
               dec_mask: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Paramters
        ---------
        dec_input: a sequence of input tokenids at current time step with shape [batch_size]
        dec_mask: mask the embedding at the current step with shape [batch_size] or None
            # target batch 3 with length 2, 1, 3 => mask = [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
            # Each time step dec_mask corresponds to each column of the mask.
            # For example: in time step 2, the second column [1, 0, 1] as the dec_mask

        Returns
        -------
        the dec_output(or presoftmax) with shape (batch_size x dec_output_size)
        the att_output of key 'p_context' with its value (batch_size x context_length)
            and key 'expected_context' with its value (batch_size x context_size).

        Note:
        Before calling self.decode, make sure the context is already set by calling self.encode.
        """
        assert dec_input.dim() == 1, "Input of decoder should with a sequence of tokenids with size [batch_size]"
        dec_input_embedding = self.dec_embedder(dec_input)
        dec_input_embedding = F.dropout(dec_input_embedding, p=self.dec_embedding_dropout, training=self.training)
        dec_output, att_output = self.decoder.decode(dec_input_embedding, dec_mask)
        return self.dec_presoftmax(dec_output), att_output

    def reset(self):
        """ Reset the decoder state.
        e.g. the luong decoder sets the attentional vector of the previous time step to be None,
        which means forgetting all the formation accumulated
        in the history (by RNN) before the current time step
        and forgetting the attention information of the previous time step.
        """
        self.decoder.reset()
