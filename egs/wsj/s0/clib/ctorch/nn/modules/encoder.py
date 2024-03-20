from typing import List, Dict, Tuple, Union, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack, pad_sequence

from clib.ctorch.utils.module_util import get_rnn, get_act
from clib.ctorch.utils.tensor_util import length2mask

class PyramidRNNEncoder(nn.Module):
    """ The RNN encoder with support of subsampling (for input with long length such as speech feature).
    https://arxiv.org/abs/1508.01211 "LAS" section 3.1 formula (5)

    The PyramidRNNEncoder accepts the feature (batch_size x max_seq_length x in_size),
    passes the feature to several layers of feedforward neural network (fnn)
    and then to several layers of RNN (rnn) with subsampling
    (by concatenating every pair of frames with type 'concat'
    or by taking the first frame every frame pair with the type 'drop').

    ps: We can pass the parameters by copying the same configuration to each layer
        or by specifying a list of configurations for each layer.
        We do padding at the end of sequence whenever subsampling needs more frames to concatenate.
    """

    def __init__(self,
                 enc_input_size: int,
                 enc_fnn_sizes: List[int] = [512],
                 enc_fnn_act: str = 'relu',
                 enc_fnn_dropout: Union[float, List[float]] = 0.25,
                 enc_rnn_sizes: List[int] = [256, 256, 256],
                 enc_rnn_config: Dict = {'type': 'lstm', 'bi': True},
                 enc_rnn_dropout: Union[float, List[float]] = 0.25,
                 enc_rnn_subsampling: Union[bool, List[bool]] = [False, True, True],
                 enc_rnn_subsampling_type: str = 'concat', # 'concat' or 'drop'
                 enc_input_padding_value: float = 0.0
    ) -> None:
        super().__init__()

        # make copy of the configuration for each layer.
        num_enc_fnn_layers = len(enc_fnn_sizes)
        if not isinstance(enc_fnn_dropout, list): enc_fnn_dropout = [enc_fnn_dropout] * num_enc_fnn_layers

        num_enc_rnn_layers = len(enc_rnn_sizes)
        if not isinstance(enc_rnn_dropout, list): enc_rnn_dropout = [enc_rnn_dropout] * num_enc_rnn_layers
        if not isinstance(enc_rnn_subsampling, list): enc_rnn_subsampling = [enc_rnn_subsampling] * num_enc_rnn_layers

        assert num_enc_fnn_layers == len(enc_fnn_dropout), \
            "Number of fnn layers does not match the lengths of specified configuration lists."
        assert num_enc_rnn_layers == len(enc_rnn_dropout) == len(enc_rnn_subsampling), \
            "Number of rnn layers does not matches the lengths of specificed configuration lists."
        assert enc_rnn_subsampling_type in {'concat', 'drop'}, \
            "The subsampling type '{}' is not implemented yet.\n".format(t) + \
            "Only support the type 'concat' and 'drop':\n" + \
            "the type 'drop' preserves the first frame every two frames;\n" + \
            "the type 'concat' concatenates the frame pair every two frames.\n"

        self.enc_input_size = enc_input_size
        self.enc_fnn_sizes = enc_fnn_sizes
        self.enc_fnn_act = enc_fnn_act
        self.enc_fnn_dropout = enc_fnn_dropout
        self.enc_rnn_sizes = enc_rnn_sizes
        self.enc_rnn_config = enc_rnn_config
        self.enc_rnn_dropout = enc_rnn_dropout
        self.enc_rnn_subsampling = enc_rnn_subsampling
        self.enc_rnn_subsampling_type = enc_rnn_subsampling_type
        self.enc_input_padding_value = enc_input_padding_value

        self.num_enc_fnn_layers = num_enc_fnn_layers
        self.num_enc_rnn_layers = num_enc_rnn_layers

        pre_size = self.enc_input_size
        self.enc_fnn_layers = nn.ModuleList()
        for i in range(num_enc_fnn_layers):
            self.enc_fnn_layers.append(nn.Linear(pre_size, enc_fnn_sizes[i]))
            self.enc_fnn_layers.append(get_act(enc_fnn_act)())
            self.enc_fnn_layers.append(nn.Dropout(p=enc_fnn_dropout[i]))
            pre_size = enc_fnn_sizes[i]

        self.enc_rnn_layers = nn.ModuleList()
        for i in range(num_enc_rnn_layers):
            rnn_layer = get_rnn(enc_rnn_config['type'])
            self.enc_rnn_layers.append(rnn_layer(pre_size, enc_rnn_sizes[i], batch_first=True, bidirectional=enc_rnn_config['bi']))
            pre_size = enc_rnn_sizes[i] * (2 if enc_rnn_config['bi'] else 1) # for bidirectional rnn
            if (enc_rnn_subsampling[i] and enc_rnn_subsampling_type == 'concat'): pre_size = pre_size * 2 # for concat subsampling

        self.output_size = pre_size

    def get_context_size(self) -> int:
        return self.output_size

    def encode(self,
               input: torch.Tensor,
               input_lengths: Union[torch.Tensor, None] = None,
               verbose: bool = False) -> (torch.Tensor, torch.Tensor):
        """ Encode the feature (batch_size x max_seq_length x in_size), optionally with its lengths (batch_size),
        and output the context vector (batch_size x max_seq_length' x context_size)
        with its mask (batch_size x max_seq_length').

        ps: the dimension of context vector is influenced by 'bidirectional' and 'subsampling (concat)' options of RNN.
            the max_seq_length' influenced by 'subsampling' options of RNN.
        """
        if (input_lengths is None):
            cur_batch_size, max_seq_length, cur_input_size = input.shape
            input_lengths = [max_seq_length] * cur_batch_size

        output = input
        for layer in self.enc_fnn_layers:
            output = layer(output)

        output_lengths = input_lengths
        for i in range(self.num_enc_rnn_layers):
            layer = self.enc_rnn_layers[i]
            # packed_sequence = pack(output, output_lengths, batch_first=True)
            packed_sequence = pack(output, output_lengths.cpu(), batch_first=True)
            output, _ = layer(packed_sequence) # LSTM returns '(output, [hn ,cn])'
            output, _ = unpack(output, batch_first=True, padding_value=self.enc_input_padding_value) # unpack returns (data, length)
            # dropout of lstm module behaves randomly even with same torch seed, so we'll append dropout layer.
            output = F.dropout(output, p=self.enc_rnn_dropout[i], training=self.training)

            # Subsampling by taking the first frame or concatenating frames for every two frames.
            if (self.enc_rnn_subsampling[i]):

                # Padding the max_seq_length be a multiple of 2 (even number) for subsampling.
                # ps: That padding frames with a multiple of 8 (with 3 times of subsampling) before inputting to the rnn
                #     equals to that padding 3 times in the middle of layers with a multiple of 2,
                #     because of the pack and unpack operation only takes feature with effective lengths to rnn.
                if (output.shape[1] % 2 != 0): # odd length
                    extended_part = output.new_ones(output.shape[0], 1, output.shape[2]) * self.enc_input_padding_value
                    output = torch.cat([output, extended_part], dim=-2) # pad to be even length

                if (self.enc_rnn_subsampling_type == 'drop'):
                    output = output[:, ::2]
                    output_lengths = torch.LongTensor([(length + (2 - 1)) // 2 for length in output_lengths]).to(output.device)
                elif (self.enc_rnn_subsampling_type == 'concat'):
                    output = output.contiguous().view(output.shape[0], output.shape[1] // 2, output.shape[2] * 2)
                    output_lengths = torch.LongTensor([(length + (2 - 1)) // 2 for length in output_lengths]).to(output.device)
                else:
                    raise NotImplementedError("The subsampling type {} is not implemented yet.\n".format(self.enc_rnn_subsampling_type) +
                                              "Only support the type 'concat' and 'drop':\n" +
                                              "The type 'drop' takes the first frame every two frames.\n" +
                                              "The type 'concat' concatenates the frame pair every two frames.\n")
            if verbose:
                print("After layer '{}' applying the subsampling '{}' with type '{}': shape is {}, lengths is {} ".format(
                    i, self.enc_rnn_subsampling[i], self.enc_rnn_subsampling_type, output.shape, output_lengths))
                print("mask of lengths is\n{}".format(length2mask(output_lengths)))

        context, context_mask = output, length2mask(output_lengths)
        return context, context_mask
