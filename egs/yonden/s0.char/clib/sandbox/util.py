# Some functions are copyed and adapted from AllenNLP; examples added.
from typing import Dict, Any, Iterable, TypeVar, List, Union, Optional, Tuple
from collections import defaultdict

import os
import sys
import re
import copy
import logging
import subprocess
import random

import numpy
import torch


try:
    import resource
except ImportError:
    # resource doesn't exist on Windows systems
    resource = None

logger = logging.getLogger(__name__)
A = TypeVar('A')


# system utilities
def get_git_commit():
    subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
    print('Git commit: {commit}')
    return commit

def get_pytorch_version() -> None:
    import torch    
    print(f"Pytorch version: {torch.__version__}")
    # logger.info(f"Pytorch version: {torch.__version__}")


def get_current_peak_memory() -> float:
    """
    Get peak memory usage for this process, as measured by
    max-resident-set size:
    https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size
    Only works on OSX and Linux, returns 0.0 otherwise.
    """
    if resource is None or sys.platform not in ('linux', 'darwin'):
        return 0.0

    # TODO(joelgrus): For whatever, our pinned version 0.521 of mypy does not like
    # next line, but later versions (e.g. 0.530) are fine with it. Once we get that
    # figured out, remove the type: ignore.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore

    if sys.platform == 'darwin':
        # On OSX the result is in bytes.
        return peak / 1_000_000

    else:
        # On Linux the result is in kilobytes.
        return peak / 1_000


def get_current_gpu_memory() -> Dict[int, int]:
    """
    Get the current GPU memory usage.
    Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    Returns
    -------
    ``Dict[int, int]``
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
        Returns an empty ``dict`` if GPUs are not available.
    """
    # pylint: disable=bare-except
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                          '--format=csv,nounits,noheader'],
                                         encoding='utf-8')
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        return {gpu: memory for gpu, memory in enumerate(gpu_memory)}
    except FileNotFoundError:
        # `nvidia-smi` doesn't exist, assume that means no GPU.
        return {}
    except:
        # Catch *all* exceptions, because this memory check is a nice-to-have
        # and we'd never want a training run to fail because of it.
        logger.exception("unable to check gpu_memory_mb(), continuing")
        return {}


def check_gpu_id(device_id: int) -> None:
    if device_id is not None and device_id >= 0:
        num_devices_available = torch.cuda.device_count()
        if num_devices_available == 0:
            raise Exception("Experiment specified a GPU but none are available;"
                                     " if you want to run on CPU use the override"
                                     " 'trainer.cuda_device=-1' in the json config file.")
        elif device_id >= num_devices_available:
            raise Exception(f"Experiment specified GPU device {device_id}"
                                     f" but there are only {num_devices_available} devices "
                                     f" available.")


def set_logger(verbose: int = 1) -> None:
    """ Set a logger to record what happened when running the program

    Parameters
    ----------
        verbose: set the logging to INFO level if verbose

    Example
    -------
    In [1]: import logging
    In [2]: from clib.comm.utils import set_logger
    In [3]: set_logger()  # call the function before calling logging
    In [4]: logging.info("hello")
    2019-01-30 22:15:24,223 (<ipython-input-6-d34c3a5ceab0>:1) INFO: hello

    Note
    ----
    Please call this function before using logging; It's better to put it at the beginning of the script.
    """
    if verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')


# general utilities
def to_basic_type(x: Any) -> Any:  # pylint: disable=invalid-name,too-many-return-statements
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
    if isinstance(x, (str, float, int, bool)):
        # x is already serializable
        return x
    elif isinstance(x, torch.Tensor):
        # tensor needs to be converted to a list (and moved to cpu if necessary)
        return x.cpu().tolist()
    elif isinstance(x, numpy.ndarray):
        # array needs to be converted to a list
        return x.tolist()
    elif isinstance(x, numpy.number):
        # NumPy numbers need to be converted to Python numbers
        return x.item()
    elif isinstance(x, dict):
        # Dicts need their values sanitized
        return {key: to_basic_type(value) for key, value in x.items()}
    elif isinstance(x, (list, tuple)):
        # Lists and Tuples need their values sanitized
        return [to_basic_type(x_i) for x_i in x]
    elif x is None:
        return "None"
    elif hasattr(x, 'to_json'):
        return x.to_json()
    else:
        raise ValueError(f"Cannot sanitize {x} of type {type(x)}. "
                         "If this is your own custom class, add a `to_json(self)` method "
                         "that returns a JSON-like object.")


def is_lazy(iterable: Iterable[A]) -> bool:
    """
    Checks if the given iterable is lazy,
    which here just means it's not a list.

    Example:
        In [314]: utils.is_lazy([1, 2, 1])
        Out[314]: False

        In [315]: utils.is_lazy(zip([1, 2, 1], [2, 3, 4]))
        Out[315]: True
    """
    return not isinstance(iterable, list)


def ensure_list(iterable: Iterable[A]) -> List[A]:
    """
    An Iterable may be a list or a generator.
    This ensures we get a list without making an unnecessary copy.

    Example:
        In [319]: utils.ensure_list([1, 2, 3])
        Out[319]: [1, 2, 3]

        In [320]: utils.ensure_list(zip([1, 2, 1], [2, 3, 4]))
        Out[320]: [(1, 2), (2, 3), (1, 4)]
    """
    if isinstance(iterable, list):
        return iterable
    else:
        return list(iterable)


def get_file_extension(path: str, dot=True, lower: bool = True):
    """
    Get the file extension.

    Example:
        In [342]: utils.get_file_extension("data.json")
        Out[342]: '.json'
    """
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


def deep_merge_dicts(preferred: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dicts, preferring values from `preferred`.

    Example:
        In [345]: d1 = {1: 'one', 2: 'two'}

        In [346]: d2 = {2: 'two', 3: 'three'}

        In [347]: d3 = {2: 'tow', 3: 'three'}

        In [348]: dh1 = {'1': d1}

        In [349]: dh2 = {'1': d2}

        In [356]: utils.deep_merge_dicts(d1, d2)
        Out[356]: {1: 'one', 3: 'three', 2: 'two'}

        In [357]: utils.deep_merge_dicts(d3, d1)
        Out[357]: {3: 'three', 1: 'one', 2: 'tow'}

        In [358]: utils.deep_merge_dicts(d1, d3)
        Out[358]: {1: 'one', 3: 'three', 2: 'two'}

        In [363]: utils.deep_merge_dicts(dh1, dh2)
        Out[363]: {'1': {1: 'one', 3: 'three', 2: 'two'}}
    """
    preferred_keys = set(preferred.keys())
    fallback_keys = set(fallback.keys())
    common_keys = preferred_keys & fallback_keys

    merged: Dict[str, Any] = {}

    for key in preferred_keys - fallback_keys:
        merged[key] = copy.deepcopy(preferred[key])
    for key in fallback_keys - preferred_keys:
        merged[key] = copy.deepcopy(fallback[key])

    for key in common_keys:
        preferred_value = preferred[key]
        fallback_value = fallback[key]

        if isinstance(preferred_value, dict) and isinstance(fallback_value, dict):
            merged[key] = deep_merge_dicts(preferred_value, fallback_value)
        else:
            merged[key] = copy.deepcopy(preferred_value)

    return merged


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


# pytorch utilities
def seed_everything(seed: int=2019) -> None:
    """
    seeding everything to make the experiment
    deterministic and reproducible.
    reference: https://github.com/pytorch/pytorch/issues/11278
    """
    random.seed(seed)  # for teacher forcing
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def get_frozen_and_tunable_parameter_names(model: torch.nn.Module) -> List:
    frozen_parameter_names = []
    tunable_parameter_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen_parameter_names.append(name)
        else:
            tunable_parameter_names.append(name)
    return [frozen_parameter_names, tunable_parameter_names]


def dump_metrics(file_path: str, metrics: Dict[str, Any], log: bool = False) -> None:
    metrics_json = json.dumps(metrics, indent=2)
    with open(file_path, "w") as metrics_file:
        metrics_file.write(metrics_json)
    if log:
        logger.info("Metrics: %s", metrics_json)


# pytorch.tensor utilities
def has_tensor(obj:Any) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.

    Note:
        For dict structure, we only check the values;
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


def move_to_device(obj, device: torch.device):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).

    Example:
        In [400]: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        In [401]: data
        {'43': [tensor([1, 2]), 54], 2: {2}}
        In [402]: utils.move_to_device(data, device)
        Out[402]: {'43': [tensor([1, 2], device='cuda:0'), 54], 2: {2}}
    """
    if device.type == "cpu" or not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device) for item in obj])
    else:
        return obj


def batch_tensor_dicts(tensor_dicts: List[Dict[str, torch.Tensor]],
                       remove_trailing_dimension: bool = False) -> Dict[str, torch.Tensor]:
    """
    Takes a list of tensor dictionaries, where each dictionary is assumed to have matching keys,
    and returns a single dictionary with all tensors with the same key batched together.

    Note: the tensors with same key should have same shape

    Parameters
    ----------
    tensor_dicts : ``List[Dict[str, torch.Tensor]]``
        The list of tensor dictionaries to batch.
    remove_trailing_dimension : ``bool``
        If ``True``, we will check for a trailing dimension of size 1 on the tensors that are being
        batched, and remove it if we find it.

    Example:
        In [429]: data
        Out[429]:
        [{'k1': tensor([1, 2]), 'k2': tensor([3, 4, 5])},
         {'k1': tensor([11, 22]), 'k2': tensor([33, 44, 55])}]

        In [430]: utils.batch_tensor_dicts(data)
        Out[430]:
        {'k1': tensor([[ 1,  2],
                     [11, 22]]),
        'k2': tensor([[ 3,  4,  5],
                    [33, 44, 55]])}
    """
    key_to_tensors: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            key_to_tensors[key].append(tensor)
    batched_tensors = {}
    for key, tensor_list in key_to_tensors.items():
        batched_tensor = torch.stack(tensor_list)
        if remove_trailing_dimension and all(tensor.size(-1) == 1 for tensor in tensor_list):
            batched_tensor = batched_tensor.squeeze(-1)
        batched_tensors[key] = batched_tensor
    return batched_tensors


def tensors_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 1e-12) -> bool:
    """
    A check for tensor equality (by value).  We make sure that the tensors have the same shape,
    then check all of the entries in the tensor for equality.  We additionally allow the input
    tensors to be lists or dictionaries, where we then do the above check on every position in the
    list / item in the dictionary.  If we find objects that aren't tensors as we're doing that, we
    just defer to their equality check.
    This is kind of a catch-all method that's designed to make implementing ``__eq__`` methods
    easier, in a way that's really only intended to be useful for tests.

    Example:
        In [570]: utils.tensors_equal(torch.Tensor([1, 2]), torch.Tensor([1., 2.]))
        Out[570]: tensor(1, dtype=torch.uint8)

        In [571]: utils.tensors_equal(torch.Tensor([1, 2]), torch.Tensor([1., 2.01]))
        Out[571]: tensor(0, dtype=torch.uint8)

        In [572]: utils.tensors_equal(torch.Tensor([1, 2]), torch.Tensor([1., 2.00000000000000000000000000001]))
        Out[572]: tensor(1, dtype=torch.uint8)
    """
    # pylint: disable=too-many-return-statements
    if isinstance(tensor1, (list, tuple)):
        if not isinstance(tensor2, (list, tuple)) or len(tensor1) != len(tensor2):
            return False
        return all([tensors_equal(t1, t2, tolerance) for t1, t2 in zip(tensor1, tensor2)])
    elif isinstance(tensor1, dict):
        if not isinstance(tensor2, dict):
            return False
        if tensor1.keys() != tensor2.keys():
            return False
        return all([tensors_equal(tensor1[key], tensor2[key], tolerance) for key in tensor1])
    elif isinstance(tensor1, torch.Tensor):
        if not isinstance(tensor2, torch.Tensor):
            return False
        if tensor1.size() != tensor2.size():
            return False
        return ((tensor1 - tensor2).abs().float() < tolerance).all()
    else:
        try:
            return tensor1 == tensor2
        except RuntimeError:
            print(type(tensor1), type(tensor2))
            raise


def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.

    In [578]: log_prob_list = torch.Tensor([0.2, 0.3]).log()

    In [579]: log_prob_list
    Out[579]: tensor([-1.6094, -1.2040])

    In [580]: utils.logsumexp(log_prob_list)
    Out[580]: tensor(-0.6931)

    In [581]: log_prob_list.exp().sum().log()
    Out[581]: tensor(-0.6931)
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.

    Example:
        In [591]: utils.get_range_vector(5, 2)
        Out[591]: tensor([0, 1, 2, 3, 4], device='cuda:2')
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def get_mask_from_lengths_of_sequences(lens_of_seqs: torch.tensor, max_len: int) -> torch.tensor:
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.
    We require ``max_length`` here instead of just computing it from the input ``sequence_lengths``
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.

    Examples:
        lens_of_seqs: [2, 2, 3], max_len: 4: -> mask: [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]

        In [451]: lens = torch.tensor([2, 2, 3])
        In [452]: utils.get_mask_from_lens_of_seqs(lens, 4)                                                                                         
        Out[452]: 
        tensor([[1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0]])

        In [453]: utils.get_mask_from_lens_of_seqs(lens, 2)                                                                                         
        Out[453]: 
        tensor([[1, 1],
                [1, 1],
                [1, 1]])
    """
    ones_seqs = lens_of_seqs.new_ones(len(lens_of_seqs), max_len)
    cumsum_ones = ones_seqs.cumsum(dim=-1)

    return (cumsum_ones <= lens_of_seqs.unsqueeze(-1)).long()


def get_lengths_from_binary_sequence_mask(mask: torch.Tensor):
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.
    Parameters
    ----------
    mask : torch.Tensor, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.
    Returns
    -------
    A torch.LongTensor of shape (batch_size,) representing the lengths
    of the sequences in the batch.

    Example:
        In [458]: mask
        Out[458]:
        tensor([[1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0]])

        In [459]: utils.get_lengths_from_binary_sequence_mask(mask)
        Out[459]: tensor([2, 2, 3])

    """
    return mask.long().sum(-1)


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    Returns
    -------
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : torch.LongTensor
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permuation_index : torch.LongTensor
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.

    Example:
        In [471]: tensor = torch.tensor([[2, 1, 0, 0], [2, 3, 3, 2], [2, 3, 1, 0]])

        In [472]: lengths = torch.tensor([2, 4, 3])

        In [473]: utils.sort_batch_by_length(tensor, lengths)
        Out[473]:
        (tensor([[2, 3, 3, 2],
                 [2, 3, 1, 0],
                 [2, 1, 0, 0]]),
         tensor([4, 3, 2]),
         tensor([2, 0, 1]),
         tensor([1, 2, 0]))
    """

    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
        raise ConfigurationError("Both the tensor and sequence lengths must be torch.Tensors.")

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


def get_final_encoder_states(encoder_outputs: torch.Tensor,
                             mask: torch.Tensor,
                             bidirectional: bool = False) -> torch.Tensor:
    """
    Given the output from a ``Seq2SeqEncoder``, with shape ``(batch_size, sequence_length,
    encoding_dim)``, this method returns the final hidden state for each element of the batch,
    giving a tensor of shape ``(batch_size, encoding_dim)``.  This is not as simple as
    ``encoder_outputs[:, -1]``, because the sequences could have different lengths.  We use the
    mask (which has shape ``(batch_size, sequence_length)``) to find the final state for each batch
    instance.
    Additionally, if ``bidirectional`` is ``True``, we will split the final dimension of the
    ``encoder_outputs`` into two and assume that the first half is for the forward direction of the
    encoder and the second half is for the backward direction.  We will concatenate the last state
    for each encoder dimension, giving ``encoder_outputs[:, -1, :encoding_dim/2]`` concatenated with
    ``encoder_outputs[:, 0, encoding_dim/2:]``.

    Example:
        In [485]: encoder_outputs = torch.tensor([[2, 20, -1, -1], [2, 3, 3, 30], [2, 3, 40, -1]]).unsqueeze(-1)

        In [486]: mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0]])

        In [487]: utils.get_final_encoder_states(encoder_outputs, mask)
        Out[487]:
        tensor([[20],
                [30],
                [40]])
    """
    # These are the indices of the last words in the sequences (i.e. length sans padding - 1).  We
    # are assuming sequences are right padded.
    # Shape: (batch_size,)
    last_word_indices = mask.sum(1).long() - 1
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    # Shape: (batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, encoder_output_dim)
    if bidirectional:
        final_forward_output = final_encoder_output[:, :(encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2):]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.
    This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    ``tensor.masked_fill((1 - mask).byte(), replace_with)``.

    Example:
        In [542]: mask = torch.Tensor([[1, 1], [1, 0]]) # '1' is through, '0' is masked; think of the binary multiplication

        In [543]: tensor = torch.Tensor([[5, 6], [7, 0]])

        In [544]: utils.replace_masked_values(tensor, mask, -1)
        Out[544]:
        tensor([[ 5.,  6.],
                [ 7., -1.]])
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill((1 - mask).byte(), replace_with)


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.Tensor):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.
    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Tensor, required.
    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = (torch.rand(tensor_for_masking.size()) > dropout_probability).to(tensor_for_masking.device)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


def masked_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorical cross-entropy loss.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1) # assume the first dim is the batch dim.
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector * mask, dim=dim)
        result = result * mask
        result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
    return result


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def masked_max(vector: torch.Tensor,
               mask: torch.Tensor,
               dim: int,
               keepdim: bool = False,
               min_val: float = -1e7) -> torch.Tensor:
    """
    To calculate max along certain dimensions on masked values
    Parameters
    ----------
    vector : ``torch.Tensor``
        The vector to calculate max, assume unmasked parts are already zeros
    mask : ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    dim : ``int``
        The dimension to calculate max
    keepdim : ``bool``
        Whether to keep dimension
    min_val : ``float``
        The minimal value for paddings
    Returns
    -------
    A ``torch.Tensor`` of including the maximum values.
    """
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


def masked_mean(vector: torch.Tensor,
                mask: torch.Tensor,
                dim: int,
                keepdim: bool = False,
                eps: float = 1e-8) -> torch.Tensor:
    """
    To calculate mean along certain dimensions on masked values
    Parameters
    ----------
    vector : ``torch.Tensor``
        The vector to calculate mean.
    mask : ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    dim : ``int``
        The dimension to calculate mean
    keepdim : ``bool``
        Whether to keep dimension
    eps : ``float``
        A small value to avoid zero division problem.
    Returns
    -------
    A ``torch.Tensor`` of including the mean values.
    """
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask.float(), dim=dim, keepdim=keepdim)
    return value_sum / value_count.clamp(min=eps)


def clone(module: torch.nn.Module, num_copies: int) -> torch.nn.ModuleList:
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.
    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.
    For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
    embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:
        - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
        - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
          query for each document)
    are valid input "vectors", producing tensors of shape:
    ``(batch_size, num_queries, embedding_dim)`` and
    ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.

    # Example:
        In [551]: matrix = torch.arange(24).reshape(2, 3, 4) # (batch_size, sequence_length, encoder_output_dim)

        In [552]: matrix[1][2] = 0 # the first sentence with length 3, the second with length 2

        In [553]: matrix
        Out[553]:
        tensor([[[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]],

                [[12, 13, 14, 15],
                 [16, 17, 18, 19],
                 [ 0,  0,  0,  0]]])

        In [559]: attention = torch.Tensor([[0.5, 0.4, 0.1], [0.1, 0.1, 0.8]]) # (batch_size, sequence_length)

        In [561]: utils.weighted_sum(matrix.float(), attention) # (batch_size, encoder_output_dim)
        Out[561]:
        tensor([[2.4000, 3.4000, 4.4000, 5.4000],
                [2.8000, 3.0000, 3.2000, 3.4000]])
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            # the first dim is batch dim, so unsqueeze from the second.
            matrix = matrix.unsqueeze(1)
            # attention has extra dimensions immediately after the batch dim.
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def viterbi_decode(tag_sequence: torch.Tensor,
                   transition_matrix: torch.Tensor,
                   tag_observations: Optional[List[int]] = None):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    tag_observations : Optional[List[int]], optional, (default = None)
        A list of length ``sequence_length`` containing the class ids of observed
        elements in the sequence, with unobserved elements being set to -1. Note that
        it is possible to provide evidence which results in degenerate labelings if
        the sequences of tags you provide as evidence cannot transition between each
        other, or those transitions are extremely unlikely. In this situation we log a
        warning, but the responsibility for providing self-consistent evidence ultimately
        lies with the user.
    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : torch.Tensor
        The score of the viterbi path.

    Example:
        # sequence of length 3 with 2 states
        In [524]: scores = torch.Tensor([[0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

        # If in state 1 always in state 1; state 2 can go to state 1 or 2
        In [525]: transition_matrix = torch.Tensor([[1, 0], [1, 1]])

        In [526]: utils.viterbi_decode(scores.log(), transition_matrix.log())
        Out[526]: ([1, 1, 1], tensor(-1.5325))

        # we can't use 0.9 * 0.7 * 0.8 because of transition matrix.
        In [527]: 0.9 * 0.3 * 0.8 > 0.9 * 0.7 * 0.2
        Out[527]: True

        In [528]: torch.Tensor([0.9, 1, 0.3, 1, 0.8]).log().sum()
        Out[528]: tensor(-1.5325)

        # transition_matrix indicates if current tag is 0, then next and the future will all be 0
        # Once we have tag_observation, the score is meaningless
        In [529]: utils.viterbi_decode(scores.log(), transition_matrix.log(), [-1, 0, -1])
        Out[529]: ([1, 0, 0], tensor(99998.3906))
    """
    sequence_length, num_tags = list(tag_sequence.size())
    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ConfigurationError("Observations were provided, but they were not the same length "
                                     "as the sequence. Found sequence of length: {} and evidence: {}"
                                     .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    path_scores = []
    path_indices = []

    # Once we have tag_observation, the score is meaningless
    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot)
    else:
        path_scores.append(tag_sequence[0, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix
        scores, paths = torch.max(summed_potentials, 0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning("The pairwise potential between tags you have passed as "
                               "observations is extremely unlikely. Double check your evidence "
                               "or transition potentials!")
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot)
        else:
            path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    viterbi_score, best_path = torch.max(path_scores[-1], 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_timestep in reversed(path_indices):
        viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
    # Reverse the backward path.
    viterbi_path.reverse()
    return viterbi_path, viterbi_score


def sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       average: str = "batch",
                                       label_smoothing: float = None) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.
    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    average: str, optional (default = "batch")
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If ``None``, return a vector
        of losses per batch element.
    label_smoothing : ``float``, optional (default = None)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
        the correct label.
    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
    If ``average is None``, the returned loss is a vector of shape (batch_size,).

    Example:
        In [659]: logits = torch.arange(24).reshape(2, 3, 4).float()

        In [660]: logits[1][2] = 0 # len(batch1) = 3, len(batch2) = 2

        In [662]: logits # shape of (batch_size, sequence_length, num_classes)
        Out[662]:
        tensor([[[ 0.,  1.,  2.,  3.],
                 [ 4.,  5.,  6.,  7.],
                 [ 8.,  9., 10., 11.]],

                [[12., 13., 14., 15.],
                 [16., 17., 18., 19.],
                 [ 0.,  0.,  0.,  0.]]])

        In [667]: targets = torch.LongTensor([[1, 2, 1], [0, 3, 1]])

        In [668]: weights = torch.FloatTensor([[1, 1, 1], [1, 1, 0]]) # mask

        In [669]: utils.sequence_cross_entropy_with_logits(logits, targets, weights)
        Out[669]: tensor(2.0235)

        # computation in details
        In [670]: torch.nn.functional.log_softmax(logits, dim = -1)
        Out[670]:
        tensor([[[-3.4402, -2.4402, -1.4402, -0.4402],
                 [-3.4402, -2.4402, -1.4402, -0.4402],
                 [-3.4402, -2.4402, -1.4402, -0.4402]],

                [[-3.4402, -2.4402, -1.4402, -0.4402],
                 [-3.4402, -2.4402, -1.4402, -0.4402],
                 [-1.3863, -1.3863, -1.3863, -1.3863]]])

        In [674]: torch.nn.functional.log_softmax(logits, dim = -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1) * weights # compute Information for each token and add mask
        Out[674]:
        tensor([[-2.4402, -1.4402, -2.4402],
                [-3.4402, -0.4402, -0.0000]])

        In [675]: (torch.nn.functional.log_softmax(logits, dim = -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1) * weights).sum(-1) / weights.sum(-1) # average over seq_len(batch1)=3 and seq_len(batch2)=2
        Out[675]: tensor([-2.1069, -1.9402])

        In [678]: -((torch.nn.functional.log_softmax(logits, dim = -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1) * weights).sum(-1) / weights.sum(-1)).sum() / 2 # averge the 2 batches
        Out[678]: tensor(2.0235)
    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss


