# kaldi_data.py implemented by bin-wu at 23:43 in 2019.01.13

from typing import List, Dict, Any, Callable, Optional, TypeVar
import logging
import collections
import json
import pprint

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence
from clib.kaldi import kaldi_io


Instance = Dict[str, Any]
T = TypeVar('T')


class KaldiDataset(Dataset):
    """The kaldi dataset is the collection of utterance instances with different fields.

    Parameters
    ----------
    instances: a list of instances, each instance contains several fields.
    field_to_sort: we use to sort all instances by the field. (assume integer field)
        For example, we sort the instances according to the field 'num_frames', as needed by seq2seq model of ASR.

    Example
    -------
    In [24]: instances = [{'uttid': '011', 'feat': 'feat/feats.1.ark:13', 'feat_dim': '83', 'num_frames': '845', 'tokenid': '20 38 18'},
                          {'uttid': '02c', 'feat': 'feat/feats.1.ark:36', 'feat_dim': '83', 'num_frames': '840', 'tokenid': '27 14'}]
    In [25]: dataset = KaldiDataset(instances, field_to_sort='num_frames')
    In [26]: for instance in dataset:
        ...:     print(instance)
        ...:
    {'uttid': '02c', 'feat': 'feat/feats.1.ark:36', 'feat_dim': '83', 'num_frames': '850', 'tokenid': '27 14'}
    {'uttid': '011', 'feat': 'feat/feats.1.ark:13', 'feat_dim': '83', 'num_frames': '845', 'tokenid': '20 38 18'}
    """

    def __init__(self,
                 instances: List[Instance],
                 field_to_sort: str = "num_frames") -> None:
        super().__init__()

        self.instances: List[Instance] = []
        # We remove instances with empty sequences, as pytorch can not deal with the empty targets.
        for instance in instances:
            if 'num_frames' in instance and int(instance['num_frames']) == 0:
                logging.warning(f"The utterance with id {instance['uttid']} has an empty frame sequence. Discard it.")
                continue
            if 'num_tokens' in instance and int(instance['num_tokens']) == 0:
                logging.warning(f"The utterance with id {instance['uttid']} has an empty token sequence. Discard it.")
                continue
            self.instances.append(instance)

        if field_to_sort:
            self._sort_by_field(field_to_sort)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> List[Instance]:
        return self.instances[index]

    def _sort_by_field(self, field_to_sort: str) -> None:
        self.instances = sorted(self.instances, key=lambda instance: int(instance[field_to_sort]), reverse=True)

    def __str__(self):
        fstring = f'The kaldi database has {self.__len__()} instances\n'
        fstring += f'Each instance has fields {self.instances[0].keys()}\n'
        if self._field_to_sort: fstring += f'The database is sorted by the field \'{self._field_to_sort}\''
        return fstring


class KaldiDataLoader(DataLoader):
    """The Kaldi dataloader specially designed to deal with kaldi fields.

    Parameters
    ----------
    dataset: dataset from which to load the data.
    batch_size: how many samples per batch to load
        (default: ``1``).
    shuffle: set to ``True`` to have the data reshuffled
        at every epoch (default: ``False``).
    sampler: defines the strategy to draw samples from
        the dataset. If specified, ``shuffle`` must be False.
    batch_sampler: like sampler, but returns a batch of
        indices at a time. Mutually exclusive with :attr:`batch_size`,
        :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
    num_workers: how many subprocesses to use for data
        loading. 0 means that the data will be loaded in the main process.
        (default: ``0``)
    collate_fn: merges a list of samples to form a mini-batch;
        default kaldi specific collate_fn
    pin_memory: If ``True``, the data loader will copy tensors
        into CUDA pinned memory before returning them.
    drop_last: set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)
    timeout: if positive, the timeout value for collecting a batch
        from workers. Should always be non-negative. (default: ``0``)
    worker_init_fn: If not ``None``, this will be called on each
        worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
        input, after seeding and before data loading. (default: ``None``)
    padding_tokenid: the padding value of tokenid field

    Example
    -------
    In [6]: instances
    [{'uttid': '02c', 'feat_dim': '83', 'num_frames': '840', 'tokenid': '27'},
     {'uttid': '011', 'feat_dim': '83', 'num_frames': '845', 'tokenid': '20 38 18'},
     {'uttid': '22b', 'feat_dim': '83', 'num_frames': '783', 'tokenid': '20 54'}]
    In [7]: dataset = KaldiDataset(instances, field_to_sort='num_frames')
    In [8]: dataloader = KaldiDataLoader(dataset, batch_size=2)
    In [9]: for batch in dataloader:
        ...:     print(batch)
        ...:
    {'uttid': ['011', '02c'], 'feat_dim': 83, 'num_frames': tensor([845, 840]), 'tokenid': tensor([[20, 38, 18], [27, -1, -1]])}
    {'uttid': ['22b'], 'feat_dim': 83, 'num_frames': tensor([783]), 'tokenid': tensor([[20, 54]])}
    """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 sampler: Optional[Sampler] = None,
                 batch_sampler: Optional[Sampler] = None,
                 num_workers: int = 0,
                 collate_fn: Optional[Callable[[List[Any]], Any]] = None,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 padding_tokenid: int = -1) -> None:
        if(not collate_fn):
            collate_fn = self._kaldi_collate
        super().__init__(dataset, batch_size, shuffle, sampler,
                         batch_sampler, num_workers, collate_fn, pin_memory, drop_last)
        self.padding_tokenid = padding_tokenid

    def _kaldi_collate(self, batch: List[Any]) -> Any:
        # If batch is a list of instances, deal with each field.
        if isinstance(self.dataset, KaldiDataset):
            batch_dict = dict()
            for key in batch[0]:
                if key == 'feat_dim' or key == 'vocab_size':
                    batch_dict[key] = int(batch[0][key])
                elif key == 'num_frames' or key == 'num_tokens':
                    l = [int(d[key]) for d in batch]
                    batch_dict[key] = default_collate(l)
                elif key == 'tokenid':
                    tokenid = [torch.LongTensor(list(map(int, d[key].split()))) for d in batch]
                    batch_dict[key] = pad_sequence(tokenid, batch_first=True, padding_value=self.padding_tokenid)
                elif key == 'feat':
                    feat = [torch.FloatTensor(kaldi_io.read_mat(d['feat'])) for d in batch]
                    batch_dict[key] = pad_sequence(feat, batch_first=True)
                else:
                    batch_dict[key] = default_collate([d[key] for d in batch])
            return batch_dict
        else:
            return default_collate(batch)
