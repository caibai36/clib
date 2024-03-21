def greedy_search(model: nn.Module,
                  source: torch.Tensor,
                  source_lengths: torch.Tensor,
                  sos_id: int,
                  eos_id: int,
                  max_dec_length: int) -> Tuple[List[torch.LongTensor], torch.LongTensor, List[torch.Tensor]]:
    """ Generate the hypothesis from source by greed search (beam search with beam_size 1)

    Parameters
    ----------
    model: an attention sequence2sequence model
    source: shape of [batch_size, source_max_length, source_size]
    source_length: shape of [batch_size]
    sos_id: id of the start of sentence token
    eos_id: id of the end of sentence token
    max_dec_length: the maximum length of the hypothesis (a sequence of tokens)
        the decoder can generate, even if eos token does not occur.

    Returns
    -------
    cropped hypothesis: a list with length [batch_size];
        each element in the batch is a sequence of tokenids exclude eos_id with shape as [hypo_lengths[i]]
    cropped lengths of hypothesis: shape [batch_size]; lengths without sos_id and eos_id
    cropped attentions of hypothesis: a list with length [batch_size],
        shape of each element as [hypos_length[i], context_length[i]]

    Example
    -------
    Input:
    token2id, first_batch, model = get_token2id_firstbatch_model()
    source, source_lengths = first_batch['feat'], first_batch['num_frames']
    sos_id, eos_id = int(token2id['<sos>']), int(token2id['<eos>']) # 2, 3
    max_dec_length = 5

    model.to(device)
    source, source_lengths = source.to(device), source_lengths.to(device)
    print(greedy_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length))
    print(greedy_search(model, source, source_lengths, sos_id, eos_id, max_dec_length))

    Output:
    (tensor([[5, 4, 3, 3, 3],
             [5, 4, 5, 4, 5],
             [5, 6, 3, 3, 3]]),
     tensor([ 3, -1,  3]),
     tensor([[[0.0187, 0.9813],
             [0.0210, 0.9790],
             [0.0193, 0.9807],
             [0.0201, 0.9799],
             [0.0201, 0.9799]],

            [[0.0057, 0.9943],
             [0.0056, 0.9944],
             [0.0050, 0.9950],
             [0.0056, 0.9944],
             [0.0050, 0.9950]],

            [[1.0000, 0.0000],
             [1.0000, 0.0000],
             [1.0000, 0.0000],
             [1.0000, 0.0000],
             [1.0000, 0.0000]]], grad_fn=<StackBackward>))
   ([tensor([5, 4]), tensor([5, 4, 5, 4, 5]), tensor([5, 6])],
     tensor([2, 5, 2]),
    [tensor([[0.0187, 0.9813], [0.0210, 0.9790]], grad_fn=<SliceBackward>),
    tensor([[0.0057, 0.9943],
            [0.0056, 0.9944],
            [0.0050, 0.9950],
            [0.0056, 0.9944],
            [0.0050, 0.9950]], grad_fn=<AliasBackward>),
    tensor([[1.], [1.]], grad_fn=<SliceBackward>)])
    """

    hypo, hypo_lengths, hypo_att = greedy_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length)
    cropped_hypo_lengths = crop_hypothesis_lengths(hypo_lengths, max_dec_length)
    return (crop_hypothesis(hypo, cropped_hypo_lengths), cropped_hypo_lengths,
            crop_attention(hypo_att, cropped_hypo_lengths, mask2length(model.decoder.context_mask)))

def crop_hypothesis_lengths(hypo_lengths, max_dec_length):
    """ Remove the eos_id from lengths of the hypothesis

    Parameters
    ----------
    hypo_lengths: shape [batch_size]
    max_dec_length: the maximum length of the decoder hypothesis

    Returns
    -------
    the lengths of hypothesis with eos_id cropped
    """
    cropped_hypo_lengths = torch.ones_like(hypo_lengths)

    batch_size = hypo_lengths.shape[0]
    for i in range(batch_size):
        if hypo_lengths[i] == -1: # reach the max length of decoder without eos_id
            cropped_hypo_lengths[i] = max_dec_length
        else:
            cropped_hypo_lengths[i] = hypo_lengths[i] - 1 # remove eos_id

    return cropped_hypo_lengths

def crop_hypothesis(hypo: torch.Tensor, hypo_lengths: torch.Tensor) -> List[torch.Tensor]:
    """ Crop each instance in the hypothesis batch with its length

    Parameters
    ----------
    hypo: [batch_size, max_trans_length]
    hypo_lengths: [batch_size]

    Returns
    -------
    a list of instances of the hypothesis batch, which might have different lengths
    """
    return [hypo[i][0:hypo_lengths[i]] for i in range(len(hypo))]


def crop_attention(attention: torch.Tensor,
                   hypo_lengths: torch.Tensor,
                   context_lengths: torch.Tensor) -> List[torch.Tensor]:
    """ corp the attention with lengths of text hypothesis and lengths with feature context.

    Parameters
    ----------
    attention: shape [batch_size, max_hypothesis_length, max_context_length]
    hypo_lengths: the lengths of decoded hypothesis with shape [batch_size, max_hypothesis_length]
    context_lengths: the lengths of the context (e.g., from context mask)  with shape [batch_size, max_context_length]

    Returns
    -------
    cropped attention list with each shape as (hypothesis_length[i], context_length[i])
    """
    return [attention[i][0:hypo_lengths[i], 0:context_lengths[i]] for i in range(len(attention))]

# general utilities
def to_basic_type(x: Any) -> Any:
    """
    Sanitize turns PyTorch and Numpy types into basic Python types so they
    can be serialized into JSON.

    Example:
        In [274]: torch_tensor = torch.tensor([2, 3])
        In [275]: numpy_list = np.array([1, 2])
        In [276]: mix_list = [torch_tensor, numpy_list]
        In [278]: utils.to_basic_type([torch_tensor, numpy_list, mix_list])
        Out[278]: [[2, 3], [1, 2], [[2, 3], [1, 2]]]
        In [292]: utils.to_basic_type({torch_tensor: mix_list, 45.666: numpy_list})
        Out[292]: {tensor([2, 3]): [[2, 3], [1, 2]], 45.666: [1, 2]}
    """
    if isinstance(x, (str, float, int, bool)): return x
    elif isinstance(x, torch.Tensor): return x.cpu().tolist()
    elif isinstance(x, np.ndarray): return x.tolist()
    elif isinstance(x, np.number): return x.item()
    elif isinstance(x, dict): return {key: to_basic_type(value) for key, value in x.items()}
    elif isinstance(x, (list, tuple)): return [to_basic_type(x_i) for x_i in x]
    elif x is None: return "None"
    elif hasattr(x, 'to_json'): return x.to_json()
    else:
        raise ValueError(f"Cannot sanitize {x} of type {type(x)}. "
                         "If this is your own custom class, add a `to_json(self)` method "
                         "that returns a JSON-like object.")


def greedy_search_torch(model: nn.Module,
                       source: torch.Tensor,
                       source_lengths: torch.Tensor,
                       sos_id: int,
                       eos_id: int,
                       max_dec_length: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
    """ Generate the transcription from source by greed search (beam search with beam_size 1)

    Parameters
    ----------
    model: an attention sequence2sequence model
    source: shape of [batch_size, source_max_length, source_size]
    source_length: shape of [batch_size]
    sos_id: id of the start of sentence token
    eos_id: id of the end of sentence token
    max_dec_length: the maximum length of the transcription (a sequence of tokens)
        the decoder can generate, even if eos token does not occur.

    Returns
    -------
    transcription: shape [batch_size, dec_length];
        each transcription is a sequence of tokenid 
        (which has no sos_id, but with eos_id if
        its length is less than max_dec_length)
    lengths of transcription: shape [batch_size]; length without sos_id but with eos_id
    attentions of transcription: shape [batch_size, dec_length, context_size]
    """
    model.reset()
    model.train(False)
    model.encode(source, source_lengths)

    batch_size = source.shape[0]

    transcription_list = [] # list of different time steps
    transcription_att_list = [] 
    transcription_lengths = source.new_ones([batch_size]).long() * -1
    pre_tokenids = source.new_ones([batch_size]).long() * sos_id
    for time_step in range(max_dec_length):
        presoftmax, dec_att = model.decode(pre_tokenids)
        cur_tokenids = presoftmax.argmax(-1) # [batch_size]
        transcription_list.append(cur_tokenids)
        transcription_att_list.append(dec_att['p_context'])

        for i in range(batch_size):
            if cur_tokenids[i] == eos_id and transcription_lengths[i] == -1:
                transcription_lengths[i] = time_step + 1
        if all(transcription_lengths != -1): break
        pre_tokenids = cur_tokenids

    transcription = torch.stack(transcription_list, dim=1) # [batch_size, dec_length]
    transcription_att = torch.stack(transcription_att_list, dim=-2) # [batch_size, dec_length, context_size]
    return transcription, transcription_lengths, transcription_att

def get_token2id_firstbatch_model():
    """ for testing """
    seed = 2020
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    token2id_file = "conf/data/test_small/token2id.txt"
    token2id = {}
    with open(token2id_file, encoding='utf8') as f:
        for line in f:
            token, tokenid = line.strip().split()
            token2id[token] = tokenid
    padding_id = token2id['<pad>'] # 1

    json_file = 'conf/data/test_small/utts.json'
    utts_json = json.load(open(json_file, encoding='utf8'))
    dataloader = KaldiDataLoader(dataset=KaldiDataset(list(utts_json.values())), batch_size=3, padding_tokenid=1)
    first_batch = next(iter(dataloader))
    feat_dim, vocab_size = first_batch['feat_dim'], first_batch['vocab_size'] # global config

    pretrained_model = "conf/data/test_small/pretrained_model/model_e2000.mdl"
    model = load_pretrained_model_with_config(pretrained_model)
    return token2id, first_batch, model

token2id, first_batch, model = get_token2id_firstbatch_model()
source, source_lengths = first_batch['feat'], first_batch['num_frames']
sos_id, eos_id = int(token2id['<sos>']), int(token2id['<eos>']) # 2, 3
max_dec_length = 5
print(greedy_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length))
# parser.add_argument('--overwrite_result', action='store_true', help="overwrite result directory or not")
# def run_from_ipython():
#     try: __IPYTHON__; return True
#     except NameError: return False
# if not os.path.exists(opts['result']):
#     os.makedirs(opts['result'])
# elif run_from_ipython():
#     logging.warning("Runnnig from ipython, turn on overwriting result while not deleting old results") # ipython has problem of overwriting result repeatedly
# elif opts['overwrite_result']:
#     import shutil
#     shutil.rmtree(opts['result'])
#     logging.warning("!!!Overwriting the result directory: '{}'".format(opts['result']))
#     os.makedirs(opts['result'])
# else:
#     raise FileExistsError("Result directory '{}' already exists".format(opts['result']))

class PyramidRNNEncoder(nn.Module):
    """ The RNN encoder with support of subsampling (for input with long length such as speech feature).
    https://arxiv.org/abs/1508.01211 "LAS" section 3.1 formula (5)

    The PyramidRNNEncoder accepts the feature (batch_size x max_seq_length x in_size),
    passes the feature to several layers of feedforward neural network (fnn)
    and then to several layers of RNN (rnn) with subsampling
    (by concatenating every pair of frames with type 'pair_concat'
    or by taking the first frame every frame pair with the type 'pair_take_first').

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
                 enc_rnn_subsampling_type: str = 'pair_concat', # 'pair_concat' or 'pair_take_first'
                 enc_input_padding_value: float = 0.0
    ) -> None:
        super().__init__()

        # make copy of the configuration for each layer.
        num_enc_fnn_layers = len(enc_fnn_sizes)
        if not isinstance(enc_fnn_dropout, list): enc_fnn_dropout = [enc_fnn_dropout] * num_enc_fnn_layers

        num_enc_rnn_layers = len(enc_rnn_sizes)
        if not isinstance(enc_rnn_dropout, list): enc_rnn_dropout = [enc_rnn_dropout] * num_enc_rnn_layers
        if not isinstance(enc_rnn_subsampling, list): enc_rnn_subsampling = [enc_rnn_subsampling] * num_enc_rnn_layers

        assert num_enc_fnn_layers == len(enc_fnn_dropout), "Number of fnn layers does not match the lengths of specified configuration lists."
        assert num_enc_rnn_layers == len(enc_rnn_dropout) == len(enc_rnn_subsampling), "Number of rnn layers does not matches the lengths of specificed configuration lists."
        assert enc_rnn_subsampling_type in {'pair_concat', 'pair_take_first'}, \
            "The subsampling type '{}' is not implemented yet.\n".format(t) + \
            "Only support the type 'pair_concat' and 'pair_take_first':\n" + \
            "the type 'pair_take_first' preserves the first frame every two frames;\n" + \
            "the type 'pair_concat' concatenates the frame pair every two frames.\n"

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
            if (enc_rnn_subsampling[i] and enc_rnn_subsampling_type == 'pair_concat'): pre_size = pre_size * 2 # for pair_concat subsampling

        self.output_size = pre_size

    def get_config(self):
        return { 'class': str(self.__class__),
                 'enc_input_size': self.enc_input_size,
                 'enc_fnn_sizes': self.enc_fnn_sizes,
                 'enc_fnn_act': self.enc_fnn_act,
                 'enc_fnn_dropout': self.enc_fnn_dropout,
                 'enc_rnn_sizes': self.enc_rnn_sizes,
                 'enc_rnn_config': self.enc_rnn_config,
                 'enc_rnn_dropout': self.enc_rnn_dropout,
                 'enc_rnn_subsampling': self.enc_rnn_subsampling,
                 'enc_rnn_subsampling_type': self.enc_rnn_subsampling_type,
                 'enc_input_padding_value': self.enc_input_padding_value}

    def encode(self,
               input: torch.Tensor,
               input_lengths: Union[torch.Tensor, None] = None)->(torch.Tensor, torch.Tensor):
        """ Encode the feature (batch_size x max_seq_length x in_size),
        and output the context vector (batch_size x max_seq_length' x context_size)
        and its mask (batch_size x max_seq_length').

        Note: the dimension of context vector is influenced by 'bidirectional' and 'subsampling (pair_concat)' options of RNN.
              the max_seq_length' influenced by 'subsampling' options of RNN.
        """
        print("Shape of the input: {}".format(input.shape))

        if (input_lengths is None):
            cur_batch_size, max_seq_length, cur_input_size = input.shape
            input_lengths = [max_seq_length] * cur_batch_size

        output = input
        for layer in self.enc_fnn_layers:
            output = layer(output)

        output_lengths = input_lengths
        for i in range(self.num_enc_rnn_layers):
            layer = self.enc_rnn_layers[i]
            packed_sequence = pack(output, output_lengths, batch_first=True)
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
                    output = torch.cat([output, extended_part], dim=1) # pad to be even length

                if (self.enc_rnn_subsampling_type == 'pair_take_first'):
                    output = output[:, ::2]
                    output_lengths = torch.LongTensor([(length + (2 - 1)) // 2 for length in output_lengths]).to(output.device)
                elif (self.enc_rnn_subsampling_type == 'pair_concat'):
                    output = output.view(output.shape[0], output.shape[1] // 2, output.shape[2] * 2)
                    output_lengths = torch.LongTensor([(length + (2 - 1)) // 2 for length in output_lengths]).to(output.device)
                else:
                    raise NotImplementedError("The subsampling type {} is not implemented yet.\n".format(self.enc_rnn_subsampling_type) +
                                              "Only support the type 'pair_concat' and 'pair_take_first':\n" +
                                              "The type 'pair_take_first' takes the first frame every two frames.\n" +
                                              "The type 'pair_concat' concatenates the frame pair every two frames.\n")

            print("After layer '{}' applying the subsampling '{}' with type '{}': shape is {}, lengths is {} ".format(
                i, self.enc_rnn_subsampling[i], self.enc_rnn_subsampling_type, output.shape, output_lengths))
            print("mask of lengths is\n{}".format(length2mask(output_lengths)))

        context, context_mask = output, length2mask(output_lengths)
        return context, context_mask

class DotProductAttention(nn.Module):
    """  Attention by dot product.
    https://arxiv.org/abs/1508.04025 "Effective MNT" section 3.1 formula (8) (dot version)

    DotProductAttention is a module that takes in a dict with key of 'query' and 'context' (alternative key of 'mask' and 'need_expected_context'),
    and returns a output dict with key ('p_context' and 'expected_context').

    It takes 'query' (batch_size [x query_length] x query_size) and 'context' (batch_size x context_length x context_size),
    returns the proportion of attention ('p_context': batch_size x context_length) the query pays to different parts of context
    and the expected context vector ('expected_context': batch_size [x query_length] x context_size)
    by taking weighted average over the context by the proportion of attention.

    Example
    -------
    Input:
    query = torch.Tensor([[3, 4], [3, 5]]) # query = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    mask = torch.ByteTensor([[1, 1],[1, 0]])
    input = {'query': query, 'context': context, 'mask': mask}

    attention = DotProductAttention(2, 2)
    output = attention(input)

    Output:
    {'p_context': tensor([[0.7311, 0.2689], [0.9933, 0.0067]]),
    'expected_context': tensor([[3.2689, 3.7311], [3.0000, 4.9933]])}
    """

    def __init__(self,
                 context_size: int,
                 query_size: int,
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

    def forward(self, input:torch.Tensor):
        if (input['query'].dim() == 2): # batch_size x query_size
            return self.forward_single_time_step(input)
        elif (input['query'].dim() == 3): # batch_size x query_length x query_size
            return self.forward_multiple_time_steps(input)
        else:
            raise NotImplementedError("The shape of the query ({}) should be (batch_size [x query_length] x query_size)".format(input['query'].shape))

    def forward_multiple_time_steps(self, input: torch.tensor) -> Dict:
        query = input['query'] # batch_size x query_length x query_size

        output_sequence = []
        for i in range(query.shape[1]):
            input_cur_step = dict(input)
            input_cur_step['query'] = query[:,i,:]
            output_sequence.append(self.forward_single_time_step(input_cur_step).values())

        # output_pair[0]: list of p_context at every time step with shape (batch_size x context_length)
        # output_pair[1]: list of expected_context at every time step with shape (batch_size x context_size)
        output_pair = list(zip(*output_sequence))
        return {'p_context': torch.stack(output_pair[0], dim=1),
                'expected_context': torch.stack(output_pair[1], dim=1)}

    def forward_single_time_step(self, input: torch.Tensor) -> Dict:
        query = input['query'] # batch_size x query_size
        context = input['context'] # batch_size x context_length x context_size
        assert query.shape[-1] == context.shape[-1], \
            "The query_size ({}) and context_size ({}) need to be same for the DotProductAttention.".format(query.shape[-1], context.shape[-1])
        mask = input.get('mask', None)
        need_expected_context = input.get('need_expected_context', True)

        # score = dot_product(context,query) formula (8) of "Effective MNT".
        score = torch.bmm(context, query.unsqueeze(-1)).squeeze(-1) # batch_size x context_length
        if self.normalize: score = score / math.sqrt(query_size)
        if mask is not None: score.masked_fill(mask==0, -1e9)
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

    It takes 'query' (batch_size [x query_length] x query_size) and 'context' (batch_size x context_length x context_size),
    returns the proportion of attention ('p_context': batch_size x context_length) the query pays to different parts of context
    and the expected context vector ('expected_context': batch_size [x query_length] x context_size)
    by taking weighted average over the context by the proportion of attention.

    Example
    -------
    In:
    query = torch.Tensor([[3, 4], [3, 5]]) # query = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    mask = torch.ByteTensor([[1, 1],[1, 0]])
    input = {'query': query, 'context': context, 'mask': mask}

    attention = MLPAttention(context.shape[-1], query.shape[-1])
    output = attention(input)

    Out:
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

    def forward(self, input:torch.Tensor):
        if (input['query'].dim() == 2): # batch_size x query_size
            return self.forward_single_time_step(input)
        elif (input['query'].dim() == 3): # batch_size x query_length x query_size
            return self.forward_multiple_time_steps(input)
        else:
            raise NotImplementedError("The shape of the query ({}) should be (batch_size [x query_length] x query_size)".format(input['query'].shape))

    def forward_multiple_time_steps(self, input: torch.tensor) -> Dict:
        query = input['query'] # batch_size x query_length x query_size

        output_sequence = []
        for i in range(query.shape[1]):
            input_cur_step = dict(input)
            input_cur_step['query'] = query[:,i,:]
            output_sequence.append(self.forward_single_time_step(input_cur_step).values())

        # output_pair[0]: list of p_context at every time step with shape (batch_size x context_length)
        # output_pair[1]: list of expected_context at every time step with shape (batch_size x context_size)
        output_pair = list(zip(*output_sequence))
        return {'p_context': torch.stack(output_pair[0], dim=1),
                'expected_context': torch.stack(output_pair[1], dim=1)}

    def forward_single_time_step(self, input: torch.Tensor) -> Dict:
        query = input['query'] # batch_size x query_size
        batch_size, query_size = query.shape
        context = input['context'] # batch_size x context_length x context_size
        batch_size, context_length, context_size = context.shape
        mask = input.get('mask', None)
        need_expected_context = input.get('need_expected_context', True)

        # score = V*tanh(W[context,query]) formula (8) of "Effective MNT".
        concat = torch.cat([context, query.unsqueeze(-2).expand(batch_size, context_length, query_size)], dim=-1) # batch_size x context_length x (context_size + query_size)
        score = self.proj2score(self.att_act(self.concat2proj(concat))).squeeze(-1) # batch_size x context_length

        if mask is not None: score.masked_fill(mask==0, -1e9)
        p_context = F.softmax(score, dim=-1)
        expected_context = self.compute_expected_context(p_context, context) if need_expected_context else None # batch_size x context_size
        return {'p_context': p_context,
                'expected_context': expected_context}

def get_rnn_config(config):
    module_type = config['type'].lower()
    module_args = config.get('args', [])
    module_kargs = config.get('kargs', {})

    if module_type == 'lstm':
        rnn = nn.LSTM
    elif module_type == 'gru':
        rnn = nn.GRU
    elif module_type == 'rnn':
        rnn = nn.RNN
    elif module_type == 'lstmcell':
        rnn = nn.LSTMCell
    elif module_type == 'grucell':
        rnn = nn.GRUCell
    elif module_type == 'rnncell':
        rnn = nn.RNNCell
    else:
        raise NotImplementedError("rnn type {} is not implemented".format(module_type))

    return rnn(*args, **kargs)


def get_act_func(name):
    """ Get the activation function by name string.
    The name be 'relu', 'leaky_relu' and etc.
    """
    if(not name):
        return lambda x:x
    elif (getattr(F, name, None)):
        return getattr(F, name)
    elif (getattr(torch, name, None)):
        return getattr(torch, name)
    else:
        raise NotImplementedError("The activation function {} is not implemented.".format(name))


def get_func2(name):
    """ Get the function by name string.
    The name be 'relu', 'leaky_relu' and etc.
    """
    if(not name):
        return lambda x:x
    elif (getattr(F, name, None)):
        return getattr(F, name)
    elif (getattr(torch, name, None)):
        return getattr(torch, name)
    else:
        # [key, str(value)] format ['relu', '<function relu at 0x7f6fb8121a70>'],
        available_func =  [key for key, value in torch.nn.functional.__dict__.items() if ("built-in " in str(value) or "function " in str(value)) and str(key)[0] != '_' and str(key)[-1] != '_']
        raise NotImplementedError("The function '{}' is not implemented.\nAvaliable functions include {}".format(name, available_func))



    avaliable_att = list(registered_att.keys())

    if name.lower() in registered_att:
        return registered_att[name.lower()]
    else:
        raise NotImplementedError("The att module '{}' is not implemented\nAvaliable att modules include {}".format(name, avaliable_att))

def length2mask(sequence_lengths: torch.Tensor, max_length: Union[int, None] = None) -> torch.Tensor:
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.

    Examples:
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

class PyramidRNNEncoder(nn.Module):
    """ The RNN encoder with support of subsampling (for input with long length such as speech feature).
    https://arxiv.org/abs/1508.01211 "LAS" section 3.1 formula (5)

    The PyramidRNNEncoder accepts the feature (batch_size x max_seq_length x in_size),
    passes the feature to several layers of feedforward neural network (fnn)
    and then to several layers of RNN (rnn) with subsampling
    (by concatenating every pair of frames with type 'pair_concat'
    or by taking the first frame every frame pair with the type 'pair_take_first').

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
                 enc_rnn_subsampling_type: str = 'pair_concat', # 'pair_concat' or 'pair_take_first'
                 enc_input_padding_value: float = 0.0
    ) -> None:
        super().__init__()

        # make copy of the configuration for each layer.
        num_enc_fnn_layers = len(enc_fnn_sizes)
        if not isinstance(enc_fnn_dropout, list): enc_fnn_dropout = [enc_fnn_dropout] * num_enc_fnn_layers

        num_enc_rnn_layers = len(enc_rnn_sizes)
        if not isinstance(enc_rnn_dropout, list): enc_rnn_dropout = [enc_rnn_dropout] * num_enc_rnn_layers
        if not isinstance(enc_rnn_subsampling, list): enc_rnn_subsampling = [enc_rnn_subsampling] * num_enc_rnn_layers

        assert num_enc_fnn_layers == len(enc_fnn_dropout), "Number of fnn layers does not match the lengths of specified configuration lists."
        assert num_enc_rnn_layers == len(enc_rnn_dropout) == len(enc_rnn_subsampling), "Number of rnn layers does not matches the lengths of specificed configuration lists."
        assert enc_rnn_subsampling_type in {'pair_concat', 'pair_take_first'}, \
            "The subsampling type '{}' is not implemented yet.\n".format(t) + \
            "Only support the type 'pair_concat' and 'pair_take_first':\n" + \
            "the type 'pair_take_first' preserves the first frame every two frames;\n" + \
            "the type 'pair_concat' concatenates the frame pair every two frames.\n"

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
            if (enc_rnn_subsampling[i] and enc_rnn_subsampling_type == 'pair_concat'): pre_size = pre_size * 2 # for pair_concat subsampling

        self.output_size = pre_size

    def get_config(self):
        return { 'class': str(self.__class__),
                 'enc_input_size': self.enc_input_size,
                 'enc_fnn_sizes': self.enc_fnn_sizes,
                 'enc_fnn_act': self.enc_fnn_act,
                 'enc_fnn_dropout': self.enc_fnn_dropout,
                 'enc_rnn_sizes': self.enc_rnn_sizes,
                 'enc_rnn_config': self.enc_rnn_config,
                 'enc_rnn_dropout': self.enc_rnn_dropout,
                 'enc_rnn_subsampling': self.enc_rnn_subsampling,
                 'enc_rnn_subsampling_type': self.enc_rnn_subsampling_type,
                 'enc_input_padding_value': self.enc_input_padding_value}

    def encode(self,
               input: torch.Tensor,
               input_lengths: Union[torch.Tensor, None] = None)->(torch.Tensor, torch.Tensor):
        """ Encode the feature (batch_size x max_seq_length x in_size),
        and output the context vector (batch_size x max_seq_length' x context_size)
        and its mask (batch_size x max_seq_length').

        Note: the dimension of context vector is influenced by 'bidirectional' and 'subsampling (pair_concat)' options of RNN.
              the max_seq_length' influenced by 'subsampling' options of RNN.
        """
        print("Shape of the input: {}".format(input.shape))

        if (input_lengths is None):
            cur_batch_size, max_seq_length, cur_input_size = input.shape
            input_lengths = [max_seq_length] * cur_batch_size

        output = input
        for layer in self.enc_fnn_layers:
            output = layer(output)

        output_lengths = input_lengths
        for i in range(self.num_enc_rnn_layers):
            layer = self.enc_rnn_layers[i]
            packed_sequence = pack(output, output_lengths, batch_first=True)
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
                    output = torch.cat([output, extended_part], dim=1) # pad to be even length

                if (self.enc_rnn_subsampling_type == 'pair_take_first'):
                    output = output[:, ::2]
                    output_lengths = torch.LongTensor([(length + (2 - 1)) // 2 for length in output_lengths]).to(output.device)
                elif (self.enc_rnn_subsampling_type == 'pair_concat'):
                    output = output.view(output.shape[0], output.shape[1] // 2, output.shape[2] * 2)
                    output_lengths = torch.LongTensor([(length + (2 - 1)) // 2 for length in output_lengths]).to(output.device)
                else:
                    raise NotImplementedError("The subsampling type {} is not implemented yet.\n".format(self.enc_rnn_subsampling_type) +
                                              "Only support the type 'pair_concat' and 'pair_take_first':\n" +
                                              "The type 'pair_take_first' takes the first frame every two frames.\n" +
                                              "The type 'pair_concat' concatenates the frame pair every two frames.\n")

            print("After layer '{}' applying the subsampling '{}' with type '{}': shape is {}, lengths is {} ".format(
                i, self.enc_rnn_subsampling[i], self.enc_rnn_subsampling_type, output.shape, output_lengths))
            print("mask of lengths is\n{}".format(length2mask(output_lengths)))

        context, context_mask = output, length2mask(output_lengths)
        return context, context_mask

def test_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))

    input = next(iter(dataloader))['feat']
    input_lengths = next(iter(dataloader))['num_frames'] # or None
    # input_lengths = torch.LongTensor([7, 2, 1])
    # input = batches[1]['feat']
    # input_lengths = batches[1]['num_frames']
    enc_input_size = input.shape[-1]

    speech_encoder = PyramidRNNEncoder(enc_input_size,
                                       enc_fnn_sizes = [4, 9],
                                       enc_fnn_act = 'ReLU',
                                       enc_fnn_dropout = 0.25,
                                       enc_rnn_sizes = [5, 5, 5],
                                       enc_rnn_config = {'type': 'lstm', 'bi': True},
                                       enc_rnn_dropout = 0.25,
                                       enc_rnn_subsampling = [False, True, True],
                                       enc_rnn_subsampling_type = 'pair_concat')
    speech_encoder.get_config()

    speech_encoder.to(device)
    input, input_lengths = input.to(device), input_lengths.to(device)
    context, context_mask = speech_encoder.encode(input, input_lengths)
    # print(context.shape, context_mask)

class DotProductAttention(nn.Module):
    """  Attention by dot product.
    https://arxiv.org/abs/1508.04025 "Effective MNT" section 3.1 formula (8) (dot version)

    DotProductAttention is a module that takes in a dict with key of 'query' and 'context' (alternative key of 'mask' and 'need_expected_context'),
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

    def forward(self, input: torch.Tensor) -> Dict:
        query = input['query'] # batch_size x query_size
        context = input['context'] # batch_size x context_length x context_size
        assert query.shape[-1] == context.shape[-1], \
            "The query_size ({}) and context_size ({}) need to be same for the DotProductAttention.".format(query.shape[-1], context.shape[-1])
        mask = input.get('mask', None)
        need_expected_context = input.get('need_expected_context', True)

        # score = dot_product(context,query) formula (8) of "Effective MNT".
        score = torch.bmm(context, query.unsqueeze(-1)).squeeze(-1) # batch_size x context_length
        if self.normalize: score = score / math.sqrt(query_size)
        if mask is not None: score.masked_fill(mask==0, -1e9)
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

    def forward(self, input: torch.Tensor) -> Dict:
        query = input['query'] # batch_size x query_size
        batch_size, query_size = query.shape
        context = input['context'] # batch_size x context_length x context_size
        batch_size, context_length, context_size = context.shape
        mask = input.get('mask', None)
        need_expected_context = input.get('need_expected_context', True)

        # score = V*tanh(W[context,query]) formula (8) of "Effective MNT".
        concat = torch.cat([context, query.unsqueeze(-2).expand(batch_size, context_length, query_size)], dim=-1) # batch_size x context_length x (context_size + query_size)
        score = self.proj2score(self.att_act(self.concat2proj(concat))).squeeze(-1) # batch_size x context_length

        if mask is not None: score.masked_fill(mask==0, -1e9)
        p_context = F.softmax(score, dim=-1)
        expected_context = self.compute_expected_context(p_context, context) if need_expected_context else None # batch_size x context_size
        return {'p_context': p_context,
                'expected_context': expected_context}

def test_attention() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))

    query = torch.Tensor([[3, 4], [3, 5]])
#    query = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    mask = torch.ByteTensor([[1, 1],[1, 0]])
    input = {'query': query.to(device), 'context': context.to(device), 'mask': mask.to(device)}

    attention = DotProductAttention(context.shape[-1], query.shape[-1])
    attention.to(device)
    output = attention(input)
    print(output)

    attention = MLPAttention(context.shape[-1], query.shape[-1])
    attention.to(device)
    output = attention(input)
    print(output)

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
                 context_proj_dropout: int = 0.25):
        super().__init__()

        # Copy the configuration for each layer
        num_rnn_layers = len(rnn_sizes)
        if not isinstance(rnn_dropout, list): rnn_dropout = [rnn_dropout] * num_rnn_layers
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
        self.rnn_layers = nn.ModuleList()
        pre_size = input_size + context_proj_size # input feeding
        for i in range(num_rnn_layers):
            self.rnn_layers.append(get_rnn(rnn_config['type'])(pre_size, rnn_sizes[i]))
            pre_size = rnn_sizes[i]

        # Get expected context vector from attention
        self.attention_layer = get_att(att_config['type'])(context_size, pre_size)

        # Combine hidden state and context vector to be attentional vector.
        self.context_proj_layer = nn.Linear(pre_size + context_size, context_proj_size)

        self.output_size = context_proj_size

    def set_context(self, context: torch.Tensor, context_mask: torch.Tensor = None):
        self.context = context
        self.context_mask = context_mask

    def reset(self):
        self.attentional_vector_pre = None

    def forward(self, input: torch.Tensor, dec_mask: torch.Tensor = None) -> (torch.Tensor, Dict):
        """
        input: batch_size x input_size
        dec_mask: batch_size
        # target batch 3 with length 2, 1, 3 => mask = [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
        # Each time step corresponds to each column of the mask.
        # In time step 2, the second column [1, 0, 1] as the dec_mask
        # dec_mask with shape [batch_size]
        # target * dec_mask.unsqueeze(-1).expend_as(target) will mask out
        # the feature of the second element of batch at time step 2, while the element with length 1
        """
        batch_size, input_size = input.shape
        if self.attentional_vector_pre is None:
            self.attentional_vector_pre = input.new_zeros(batch_size, self.context_proj_size)

        # Input feeding: initialize the input of LSTM with previous attentional vector information
        output = torch.cat([input, self.attentional_vector_pre], dim=-1)
        for i in range(self.num_rnn_layers):
            output, _ = self.rnn_layers[i](output) # LSTM cell return (h, c)
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

def test_luong_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))

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
    input_embedding, context, context_mask = input_embedding.to(device), context.to(device), context_mask.to(device)
    luong_decoder.to(device)
    luong_decoder.set_context(context, context_mask)
    output, att_out = luong_decoder(input_embedding)
    # output, att_out = luong_decoder(input_embedding, dec_mask=torch.Tensor([0, 1])) # mask the first instance in two batches
    print("output of Luong decoder: {}".format(output))
    print("output of attention layer: {}".format(att_out))

class EncRNNDecRNNAtt(nn.Module):
    """ Sequence-to-sequence module with RNN encoder and RNN decoder with attention mechanism.
    
    The default encoder is pyramid RNN Encoder (https://arxiv.org/abs/1508.01211 "LAS" section 3.1 formula (5)),
    which passes the input feature through forward feedback neural network ('enc_fnn_*') and RNN ('enc_rnn_*').
    Between the RNN layers we use the subsampling ('enc_rnn_subsampling_*') by concatenating both frames or taking the first frame every two frames.
    
    The default decoder is Luong Decoder (https://arxiv.org/abs/1508.04025 "Effective MNT"  
    section 3 formula (5) to create attentional vector; section 3.3 the input feeding approach)
    which passes the embedding of the input ('dec_embedding_*') along with previous attentional vector into a stacked RNN layers ('dec_rnn_*'), 
    linearly projects the top RNN hidden state and expected context vector to the current attentional vector ('dec_context_proj_*'),
    and feed the attentional vector to next time step.

    The default attention is the multilayer perception attention by concatenation (https://arxiv.org/abs/1508.04025 "Effective MNT" section 3.1 formula (8) (concat version))
    which concatenates the hidden state from decoder and each part of context from encoder and passes them to two layers neural network to get the alignment score.
    The scores are then normalized to get the probability of different part of context and get the expected context vector.
    
    We tie the embedding weight by default (https://arxiv.org/abs/1608.05859 "Using the Output Embedding to Improve Language Models" introduction),
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
                 enc_rnn_subsampling_type: str = 'pair_concat', # 'pair_concat' or 'pair_take_first'
                 dec_embedding_size: int = 256,
                 dec_embedding_dropout: float = 0.25,
                 dec_embedding_weights_tied: bool = True, # share weights between (input_onehot => input_embedding) and (output_embedding => pre_softmax)
                 dec_rnn_sizes: List[int] = [512, 512],
                 dec_rnn_config: Dict = {"type": "lstmcell"},
                 dec_rnn_dropout: Union[List[float], float] = 0.25,
                 dec_context_proj_size: int = 256, # the size of attentional vector
                 dec_context_proj_act: str = 'tanh',
                 dec_context_proj_dropout: int = 0.25,
                 enc_config: Dict = {'type': 'pyramid_rnn_encoder'},
                 dec_config: Dict = {'type': 'luong_decoder'},
                 att_config: Dict = {'type': 'mlp'}, # configuration of attention
    ):
        super().__init__()

        assert enc_config['type'] == 'pyramid_rnn_encoder', \
            "'{}' is not implemented. Supported encoder types include 'pyramid_rnn_encoder'.".format(enc_config['type'])
        assert dec_config['type'] == 'luong_decoder', \
            "'{}' is not implemented. Supported decoder types include 'luong_encoder'.".format(dec_config['type'])

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

        # Embedder
        self.dec_embedder = nn.Linear(dec_input_size, dec_embedding_size)
        
        # Decoder
        self.decoder = LuongDecoder(att_config={"type": "mlp"},
                                    context_size=context.shape[-1],
                                    input_size=input_embedding.shape[-1],
                                    rnn_sizes=[512, 512],
                                    rnn_config={"type": "lstmcell"},
                                    rnn_dropout=0.25,
                                    context_proj_size=3, 
                                    context_proj_act='tanh')
   
    def get_config(self):
        return { 'class': str(self.__class__),
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

