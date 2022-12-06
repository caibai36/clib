import torch
from torch import nn
from torch.nn import functional as F

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
