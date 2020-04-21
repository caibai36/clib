import torch
from torch import nn
from torch.nn import functional as F

import clib

def get_rnn(name):
    """ Get the RNN module by its name string.
    We can write module name directly in the configuration file.
    We can also manage the already registered rnns or add the new custom rnns.
    The name can be "LSTM", 'lstm' and etc.

    Example
    -------
    In [1]: lstm = get_rnn('lstm')(2, 5)
    In [2]: result, _ = lstm(torch.Tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]))
    In [3]: result.shape
    Out[3]: torch.Size([2, 3, 5])
    """
    registered_rnn = {'lstm': nn.LSTM,
                      'gru': nn.GRU,
                      'rnn': nn.RNN,
                      'lstmcell': nn.LSTMCell,
                      'grucell': nn.GRUCell,
                      'rnncell': nn.RNNCell}

    avaliable_rnn = list(registered_rnn.keys())

    if name.lower() in registered_rnn:
        return registered_rnn[name.lower()]
    else:
        raise NotImplementedError("The rnn module '{}' is not implemented\nAvaliable rnn modules include {}".format(name, avaliable_rnn))

def get_act(name):
    """ Get the activation module by name string.
    The name be 'ReLU', 'leaky_relu' and etc.

    Example
    -------
    In [1]: relu = get_act('relu')()
    In [2]: relu(torch.Tensor([-1, 2]))
    Out[2]: tensor([0., 2.])
    """
    registered_act = {"relu": torch.nn.ReLU,
                      "relu6": torch.nn.ReLU6,
                      "elu": torch.nn.ELU,
                      "prelu": torch.nn.PReLU,
                      "leaky_relu": torch.nn.LeakyReLU,
                      "threshold": torch.nn.Threshold,
                      "hardtanh": torch.nn.Hardtanh,
                      "sigmoid": torch.nn.Sigmoid,
                      "tanh": torch.nn.Tanh,
                      "log_sigmoid": torch.nn.LogSigmoid,
                      "softplus": torch.nn.Softplus,
                      "softshrink": torch.nn.Softshrink,
                      "softsign": torch.nn.Softsign,
                      "tanhshrink": torch.nn.Tanhshrink}

    avaliable_act = list(registered_act.keys())

    if name.lower() in registered_act:
        return registered_act[name.lower()]
    else:
        raise NotImplementedError("The act module '{}' is not implemented\nAvaliable act modules include {}".format(name, avaliable_act))

def get_att(name):
    """ Get attention module by name string.
    The name can be 'dot_product', 'mlp' and etc.

    Example
    -------
    In [347]: query = torch.Tensor([[3, 4], [3, 5]])
    In [348]: context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    In [349]: mask = torch.ByteTensor([[1, 0],[1, 1]])
    In [350]: input = {'query': query, 'context': context, 'mask': mask}
    In [351]: attention = get_att('mlp')(context.shape[-1], query.shape[-1])
    In [353]: attention(input)
    Out[353]:{'p_context': tensor([[0.4973, 0.5027], [0.5185, 0.4815]], grad_fn=<SoftmaxBackward>),
       'expected_context': tensor([[3.5027, 3.4973], [3.0000, 4.5185]], grad_fn=<SqueezeBackward1>)}
    """
    registered_att = {'dot_product': clib.ctorch.nn.modules.attention.DotProductAttention,
                      'mlp': clib.ctorch.nn.modules.attention.MLPAttention}

    avaliable_att = list(registered_att.keys())

    if name.lower() in registered_att:
        return registered_att[name.lower()]
    else:
        raise NotImplementedError("The att module '{}' is not implemented\nAvaliable att modules include {}".format(name, avaliable_att))

def get_optim(name):
    """ Get optimizer by name string.
    The name can be 'adam', 'sgd' and etc.

    Example
    -------
    In [350]: model=nn.Linear(2, 3)
    In [351]: optimizer = get_optim('adam')(model.parameters(), lr=0.005)
    """
    registered_optim = {"adam": torch.optim.Adam,
                        "sgd": torch.optim.SGD,
                        "adamw": torch.optim.AdamW,
                        "sparse_adam": torch.optim.SparseAdam,
                        "adagrad": torch.optim.Adagrad,
                        "adadelta": torch.optim.Adadelta,
                        "rmsprop": torch.optim.RMSprop,
                        "adamax": torch.optim.Adamax,
                        "averaged_sgd": torch.optim.ASGD}

    avaliable_optim = list(registered_optim.keys())

    if name.lower() in registered_optim:
        return registered_optim[name.lower()]
    else:
        #raise NotImplementedError("The optim module '{}' is not implemented\n".format(name) +
        #                          "Avaliable optim modules include {}".format(avaliable_optim))
        raise NotImplementedError(f"The optim module '{name}' is not implemented\n"
                                  f"Avaliable optim modules include {avaliable_optim}")
