from typing import List, Dict, Tuple, Union, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from clib.ctorch.utils.tensor_util import mask2length
from clib.ctorch.nn.search.utils import crop_hypothesis_lengths

class Node:
    def __init__(self,
                 state: List[int],
                 tokenid_path: List[torch.LongTensor],
                 log_prob: torch.Tensor,
                 score: torch.Tensor,
                 coeff_length_penalty: float = 1.0,
                 active: bool = True) -> None:
        """ The node of the search tree of the beam search.

        Parameters
        ----------
        state: the path of the input indices of different time steps
        tokenid_path: the path of tokenids of output to current node; a list of LongTensors with shape [1]
        log_prob: log probability of the path;  tensors with shape [1] (e.g. torch.Tensor([0.7]))
        score: the score of the path; tensor with the shape [1], usually same as log_prob
        coeff_length_penalty: large coefficient of length penalty will increase more penalty for long sentences.
            Google NMT: https://arxiv.org/pdf/1609.08144.pdf
            formula (14): length_penalty(seq)=(5+len(seq))^coeff/(5+1)^coeff
        active: current node hit the tokenid of end of sequence (<sos>) or not

        Example
        -------
        init_node = Node(state=[4], tokenid_path=[torch.LongTensor([5])], log_prob = torch.FloatTensor([-11]), score = torch.FloatTensor([-11]))
        expanded_nodes = init_node.expand(2, torch.Tensor([0, 0.25, 0.05, 0.3, 0.4]).log(), eos_id=3, expand_size=3)
        print(init_node)
        print("---")
        for node in expanded_nodes: print(node)
        # output:
        # [score: -11.000, tokenid_path:[5], state: [4], active: True]
        # ---
        # [score: -11.916, tokenid_path:[5, 4], state: [4, 2], active: True]
        # [score: -12.204, tokenid_path:[5, 3], state: [4, 2], active: False]
        # [score: -12.386, tokenid_path:[5, 1], state: [4, 2], active: True]
        """
        self.state = state
        self.tokenid_path = tokenid_path
        self.log_prob = log_prob
        self.score = score
        self.coeff_length_penalty = coeff_length_penalty
        self.active = active

    def length_penalty(self, length:int, alpha:float=1.0, const:float=5) -> float:
        """
        Generating the long sequence will add more penalty.
        Google NMT: https://arxiv.org/pdf/1609.08144.pdf
        formula (14): lp(Y)=(5+|Y|)^alpha / (5+1)^alpha
        alpha increases => length_penalty increases
        """
        return ((const + length)**alpha) / ((const + 1)**alpha)

    def expand(self, state_t:int, log_prob_t:torch.Tensor, eos_id:int, expand_size:int=5) -> List['Node']:
        """
        Parameters
        ----------
        state_t: a index of current input (input comes from active nodes of all trees of the previous time step).
        log_prob_t: log probability for expansion of the given node. shape [vocab_size]
        expand_size: the number of the nodes one node of the previous level allows to expand to the current level.
            Usually it is equals to beam_size.
            Alternatively a number smaller than `beam_size` may give better results,
            as it can introduce more diversity into the search.
            See [Beam Search Strategies for Neural Machine Translation.
            Freitag and Al-Onaizan, 2017](https://arxiv.org/abs/1702.01806).

        Returns
        -------
        a list of candidate nodes expanded from the given node
        """
        if expand_size >= len(log_prob_t):
            expand_size = len(log_prob_t) # expand all possible active nodes
        topk_log_prob_t, topk_log_prob_t_indices = log_prob_t.topk(expand_size, dim=0) # shape [expand_size], [expand_size]
        log_seq_prob = self.log_prob + topk_log_prob_t # shape [expand_size]
        scores = log_seq_prob

        scores = scores / self.length_penalty(len(self.tokenid_path), alpha=self.coeff_length_penalty) # shape [expand_size]

        expanded_nodes = []
        for i in range(expand_size):
            active = False if topk_log_prob_t_indices[i].item() == eos_id else True
            expand_node = Node(self.state + [state_t], # e.g. [1, 3] + [4] = [1, 3, 4]
                               self.tokenid_path + [topk_log_prob_t_indices[i].unsqueeze(-1)], # [tensor([1]),tensor([0])]+[tensor([2])]=[tensor([1]),tensor([0]),tensor([2])]
                               log_seq_prob[i].unsqueeze(-1), # e.g. torch.Tensor([0.4])
                               scores[i].unsqueeze(-1),
                               coeff_length_penalty=self.coeff_length_penalty,
                               active=active)
            expanded_nodes.append(expand_node)

        return expanded_nodes

    def __repr__(self):
        return "[score: {:.3f}, tokenid_path:{}, state: {}, active: {}]".format(self.score.item(), [x.item() for x in self.tokenid_path], self.state, self.active)

class Level:
    def __init__(self,
                 beam_size: int  = 5,
                 nbest: int = 1) -> None:
        """ A level with a stack of nodes of one search tree at a certain time step for beam search

        Paramters
        ---------
        beam_size: the number of nodes at previous level allows to expand to the current level
        nbest: the number of best sequences needed to be searched.
            The nbest should be smaller or equals to the beam size
        stack: a set of nodes at current level.
            A node is active if its tokenid_path hasn't ended with <sos>, else the node is finished.
            If the number of finished nodes is more than or equals to nbest, we think this level is finished.

        Example
        -------
        print("---init---")
        batch_size = 2
        empty_nodes = [Node(state=[], tokenid_path=[], log_prob = torch.FloatTensor([0]), score = torch.FloatTensor([0])) for _ in range(batch_size)]
        batch_level = [Level(beam_size=2, nbest=2) for _ in range(batch_size)]
        for b in range(batch_size): batch_level[b].add_node(empty_nodes[b])
        for b in range(batch_size): print(f"Tree {b}, level -1: {batch_level[b]}")
        print("---time step 0---")
        if not batch_level[0].is_finished(): batch_level[0].step([0], torch.Tensor([[0.4, 0.35, 0.25]]).log(), eos_id=0, expand_size=2) # find one sequence with <eos>;tree 0 finish if nbest=1
        if not batch_level[1].is_finished(): batch_level[1].step([1], torch.Tensor([[0.2, 0.5, 0.3]]).log(), eos_id=0, expand_size=2)
        for b in range(batch_size): print(f"Tree {b}, level 0: {batch_level[b]}")
        print("---time step 1---")
        if not batch_level[0].is_finished(): batch_level[0].step([0], torch.Tensor([[0.25, 0.35, 0.4]]).log(), eos_id=0, expand_size=2)
        if not batch_level[1].is_finished(): batch_level[1].step([1, 2], torch.Tensor([[0.1, 0.8, 0.1],[0.9, 0.05, 0.05]]).log(), eos_id=0, expand_size=2)
        for b in range(batch_size): print(f"Tree {b}, level 1: {batch_level[b]}")
        # ---init---
        # Tree 0, level -1: [score: 0.000, tokenid_path:[], state: [], active: True]
        # Tree 1, level -1: [score: 0.000, tokenid_path:[], state: [], active: True]
        # ---time step 0---
        # Tree 0, level 0: [score: -1.100, tokenid_path:[0], state: [0], active: False],[score: -1.260, tokenid_path:[1], state: [0], active: True]
        # Tree 1, level 0: [score: -0.832, tokenid_path:[1], state: [1], active: True],[score: -1.445, tokenid_path:[2], state: [1], active: True]
        # ---time step 1---
        # Tree 0, level 1: [score: -1.100, tokenid_path:[0], state: [0], active: False],[score: -1.966, tokenid_path:[1, 2], state: [0, 0], active: True]
        # Tree 1, level 1: [score: -0.916, tokenid_path:[1, 1], state: [1, 1], active: True],[score: -1.309, tokenid_path:[2, 0], state: [1, 2], active: False]
        """
        assert beam_size >= nbest, "beam_size should be more or equals to nbest"
        self.beam_size = beam_size
        self.nbest = nbest
        self.stack = []

    def add_node(self, node: Node) -> None:
        self.stack.append(node)

    def get_active_nodes(self) -> List[Node]:
        """ Get a list active node at the current level. """
        return [node for node in self.stack if node.active]

    def get_finished_nodes(self) -> List[Node]:
        """ Get a list of finished nodes at the current level. """
        return [node for node in self.stack if not node.active]

    def is_finished(self) -> bool:
        """ Check whether the nbest nodes finished """
        return all([not node.active for node in self.stack[0: self.nbest]])

    def step(self, state_t:List[int], log_prob_t:torch.Tensor, eos_id:int, expand_size=5) -> None:
        """
        One step of a search tree from the previous level to the current level

        Paramters
        ---------
        state_t: indices of current input (previous active nodes of all search trees) of the search tree.
            len(state_t) = num_active_nodes_prev_level_of_search_tree
        log_prob_t: log probability vectors of current output for the search tree
            shape [num_active_nodes_prev_level_of_search_tree, dec_output_size]
        eos_id: the tokenid of the <eos> (end of sequence)
        expand_size: the number of the nodes one node of the previous level allows to expand to the current level.
        """
        nodes_next_level = []

        # Collect the expanded nodes from active nodes of the previous level
        active_nodes = self.get_active_nodes()
        num_active_nodes = len(state_t)
        assert(len(active_nodes) == num_active_nodes)
        for i in range(num_active_nodes):
            nodes_next_level.extend(active_nodes[i].expand(state_t[i], log_prob_t[i], eos_id, expand_size))

        # Collect the finished nodes of the previous level
        nodes_next_level.extend(self.get_finished_nodes())

        # prune the stack to beam_size by preserving the nodes with highest score
        self.stack = nodes_next_level
        self.sort()

    def sort(self):
        """ sort the nodes at current level and keep the nodes with highest score """
        # take all possible nodes if beam_size is very large
        beam_size = self.beam_size if self.beam_size <= len(self.stack) else len(self.stack)
        self.stack = sorted(self.stack, key = lambda node: node.score.item(), reverse=True)[0:beam_size]

    def __repr__(self):
        return ",".join(map(str, self.stack))

    @staticmethod
    def split_indices(batch_num_active_nodes: List[int]) -> Tuple[List, List]:
        """ Split indices of active nodes of all trees of the previous time step (the current input indices) for each tree

        Example
        -------
        batch_num_active_nodes: [1, 1, 1]
        starts: [0, 1, 2]
        ends:   [1, 2, 3]
        batch_num_active_nodes: [2, 0, 4]
        starts: [0, 2, 2]
        ends:   [2, 2, 6]
        """
        starts, ends = [] , []
        start = 0
        for num_active_nodes in batch_num_active_nodes:
            end = start + num_active_nodes
            starts.append(start)
            ends.append(end)
            start = end
        return starts, ends

    @staticmethod
    def get_encoder_indices(batch_num_active_nodes: List[int]) -> List[int]:
        """ Get the batch index of context for each active node.

        Examples
        --------
        batch_num_active_nodes of [1, 1, 1] output [0, 1, 2]
        batch_num_active_nodes of [2, 1, 4] output [0, 0, 1, 2, 2, 2, 2]
        batch_num_active_nodes of [2, 0, 4] output [0, 0, 2, 2, 2, 2]
        batch_num_active_nodes of [0, 1, 0] output [1]
        """
        result = []
        batch_size = len(batch_num_active_nodes)
        for i in range(batch_size):
            result.extend([i]*batch_num_active_nodes[i])
        return result

def beam_search_torch(model: nn.Module,
                      source: torch.Tensor,
                      source_lengths: torch.Tensor,
                      sos_id: int,
                      eos_id: int,
                      max_dec_length: int,
                      beam_size: int = 5,
                      expand_size: int = 5,
                      coeff_length_penalty: float = 1,
                      nbest: int = 1) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
    """ Generate the hypothesis from source by beam search.
    The beam search is a greedy algorithm to explore the search tree by expanding the most promising `beam_size' nodes
    at each level of the tree (each time step) to get the most possible sequence.
    This is the batch version of the beam search. The searching strategy applies to a batch of search trees independently.
    However, at each time step, we combine the all active nodes (the nodes without hitting the <sos> token)
    of the previous time step as the current input (or an active batch) to the model for efficiency.
    Note that the decoder (model state) is indexed by the active batch, the encoder (context) is indexed by the batch.

    Parameters
    ----------
    model: an attention sequence2sequence model
    source: shape of [batch_size, source_max_length, source_size]
    source_length: shape of [batch_size]
    sos_id: id of the start of sequence token, to create the start input
    eos_id: id of the end of sequence token, to judge a node is active or not
    max_dec_length: the maximum length of the hypothesis (a sequence of tokens)
        the decoder can generate, even if eos token does not occur.
    beam_size:   the number of the nodes all nodes of the previous level allows to expand to the current level
    expand_size: the number of the nodes one node of the previous level allows to expand to the current level
        Usually expand_size equals beam_size.
        Alternatively a number smaller than `beam_size` may give better results,
        as it can introduce more diversity into the search.
        See [Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017](https://arxiv.org/abs/1702.01806).
    coeff_length_penalty: large coefficient of length penalty will increase more penalty for long sentences.
        Google NMT: https://arxiv.org/pdf/1609.08144.pdf
        formula (14): length_penalty(seq)=(5+len(seq))^coeff/(5+1)^coeff
    nbest: get nbest sequences by the beam search

    Returns
    -------
    hypothesis: shape [batch_size * nbest, dec_length]
        each hypothesis is a sequence of tokenid, ordered as the first nbest chunk,
        the second nbest chunk, ... the batch_size-th nbest chunk
        (which has no sos_id, but with eos_id if its length <  max_dec_length)
    lengths of hypothesis: shape [batch_size * nbest]
        length without sos_id but with eos_id
    attentions of hypothesis: shape [batch_size * nbest, dec_length, context_size]
    presoftmax of hypothesis: shape [batch_size * nbest, dec_length, dec_output_size]

    References
    ----------
    Wiki beam search: https://en.wikipedia.org/wiki/Beam_search
    Basic beam search example: https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
    """
    model.reset()
    model.train(False)
    model.encode(source, source_lengths) # set the context for decoding at the same time
    context, context_mask = model.decoder.get_context_and_its_mask()

    batch_size = source.shape[0]

    cur_tokenids = source.new_full([batch_size], sos_id).long() # current input [active_batch_size]

    # Initialize a batch of search trees.
    trees = []
    for b in range(batch_size):
        empty_node = Node(state=[], tokenid_path=[], log_prob = torch.FloatTensor([0]).to(source.device),
                          score = torch.FloatTensor([0]).to(source.device), coeff_length_penalty=coeff_length_penalty, active=True)
        level = Level(beam_size, nbest)
        level.add_node(empty_node)
        trees.append(level) # a tree represented by its level at each time step
    batch_num_active_nodes = [1 for _ in range(batch_size)]

    # Explore the nbest sequences of search trees.
    att_list = []
    presoftmax_list = []
    for time_step in range(max_dec_length):
        presoftmax, dec_att = model.decode(cur_tokenids) # shape [active_batch_size, dec_output_size], [active_batch_size, context_length]
        log_prob = F.log_softmax(presoftmax, dim=-1) # shape [active_batch_size, dec_output_size]
        att_list.append(dec_att['p_context'])
        presoftmax_list.append(presoftmax)

        # Expand previous active nodes independently for each tree in the batch.
        starts, ends = Level.split_indices(batch_num_active_nodes) # previous global active indices for each tree: [2,0,4]=>starts:[0,2,2];ends:[2,2,6]
        active_nodes_all_trees = []
        for b in range(batch_size):
            if trees[b].is_finished(): continue # batch_num_active_nodes[b] = 0 by default even skipped
            state_t = list(range(starts[b], ends[b])) # length: [num_nodes_to_expand_curr_tree]
            log_prob_t = log_prob[starts[b]: ends[b]] # shape: [num_nodes_to_expand_curr_tree, dec_output_size]
            trees[b].step(state_t, log_prob_t, eos_id, expand_size=expand_size)
            active_nodes = trees[b].get_active_nodes() # active nodes current level (current time step)
            batch_num_active_nodes[b] = len(active_nodes) if not trees[b].is_finished() else 0
            if not trees[b].is_finished(): active_nodes_all_trees.extend(active_nodes)

        # print(f"------time step {time_step}------")
        # print("\n".join(map(str, trees)))
        if all([tree.is_finished() for tree in trees]): break

        # Collect the active nodes of all trees at current level for the future expansion
        cur_tokenids = torch.cat([node.tokenid_path[-1] for node in active_nodes_all_trees]) # shape [active_batch_size]
        # Update the state of decoder (e.g. attentional_vector_pre for LuongDecoder) for active nodes
        if model.decoder.__class__.__name__ == 'LuongDecoder':
            input_indices = source.new([node.state[-1] for node in active_nodes_all_trees]).long() # shape [active_batch_size] input indices for active nodes
            model.decoder.attentional_vector_pre = torch.index_select(model.decoder.attentional_vector_pre, dim=0, index=input_indices)
        # Update the state of encoder (the context) for active nodes
        context_indices = source.new(Level.get_encoder_indices(batch_num_active_nodes)).long() # get the batch index of context for each active node: [2,0,4]=>[0,0,2,2,2,2]
        model.decoder.set_context(context.index_select(dim=0, index=context_indices), \
                                  context_mask.index_select(dim=0, index=context_indices) if context_mask is not None else None)

    # Generate the hypothesis from the last level of the tree
    hypo_list = [] # list of different time steps
    hypo_length_list = []
    hypo_att_list = []
    hypo_presoftmax_list = []
    for b in range(batch_size):
        for n in range(nbest):
            node = trees[b].stack[n] # iterate over nbest all active nodes at the last level of the tree
            hypo_list.append(torch.cat(node.tokenid_path))
            hypo_length_list.append(len(node.tokenid_path) if node.tokenid_path[-1].item() == eos_id else -1) # -1 means not finished

            node_att_list = [att_list[t][node.state[t]] for t in range(len(node.state))]
            node_att = torch.stack(node_att_list) # shape [dec_length, context_size]
            hypo_att_list.append(node_att)

            node_presoftmax_list = [presoftmax_list[t][node.state[t]] for t in range(len(node.state))]
            node_presoftmax = torch.stack(node_presoftmax_list) # shape [dec_length, dec_output_size]
            hypo_presoftmax_list.append(node_presoftmax)

    hypo = pad_sequence(hypo_list, batch_first=True, padding_value=eos_id)
    hypo_lengths = source.new(hypo_length_list).long()
    hypo_att = pad_sequence(hypo_att_list, batch_first=True, padding_value=0)
    hypo_presoftmax = pad_sequence(hypo_presoftmax_list, batch_first=True, padding_value=0)

    # recover the state of the model
    model.decoder.set_context(context, context_mask)
    model.reset()
    return hypo, hypo_lengths, hypo_att, hypo_presoftmax

def beam_search(model: nn.Module,
                source: torch.Tensor,
                source_lengths: torch.Tensor,
                sos_id: int,
                eos_id: int,
                max_dec_length: int,
                beam_size: int = 5,
                expand_size: int = 5,
                coeff_length_penalty: float = 1,
                nbest: int = 1) -> Tuple[List[torch.LongTensor], torch.LongTensor, List[torch.Tensor], List[torch.Tensor]]:
    """ Generate the hypothesis from source by beam search.
    The beam search is a greedy algorithm to explore the search tree by expanding the most promising `beam_size' nodes
    at each level of the tree (each time step) to get the most possible sequence.
    This is the batch version of the beam search. The searching strategy applies to a batch of search trees independently.
    However, at each time step, we combine the all active nodes (the nodes without hitting the <sos> token)
    of the previous time step as the current input (or an active batch) to the model for efficiency.
    Note that the decoder (model state) is indexed by the active batch, the encoder (context) is indexed by the batch.

    Parameters
    ----------
    model: an attention sequence2sequence model
    source: shape of [batch_size, source_max_length, source_size]
    source_length: shape of [batch_size]
    sos_id: id of the start of sequence token, to create the start input
    eos_id: id of the end of sequence token, to judge a node is active or not
    max_dec_length: the maximum length of the hypothesis (a sequence of tokens)
        the decoder can generate, even if eos token does not occur.
    beam_size:   the number of the nodes all nodes of the previous level allows to expand to the current level
    expand_size: the number of the nodes one node of the previous level allows to expand to the current level
        Usually expand_size equals beam_size.
        Alternatively a number smaller than `beam_size` may give better results,
        as it can introduce more diversity into the search.
        See [Beam Search Strategies for Neural Machine Translation.
            Freitag and Al-Onaizan, 2017](https://arxiv.org/abs/1702.01806).
    coeff_length_penalty: large coefficient of length penalty will increase more penalty for long sentences.
        Google NMT: https://arxiv.org/pdf/1609.08144.pdf
        formula (14): length_penalty(seq)=(5+len(seq))^coeff/(5+1)^coeff
    nbest: get nbest sequences by the beam search

    Returns
    -------
    cropped hypothesis: a list of [hypo_lengths[i]] tensors with the length batch_size*nbest
        each element in the batch is a sequence of tokenids excluding eos_id.
        ordered as the first nbest chunk, the second nbest chunk, ... the batch_size-th nbest chunk
    cropped lengths of hypothesis: shape [batch_size]; excluding sos_id and eos_id
    cropped attentions of hypothesis: a list of [hypo_lengths[i], context_length[i]] tensors
        with the length batch_size*nbest
    cropped presoftmax of hypothesis: a list of [hypo_lengths[i], dec_output_size] tensors
        with the lenght batch_size*nbest (hypo can not back propagate, but hypo presoftmax can)

    References
    ----------
    Wiki beam search: https://en.wikipedia.org/wiki/Beam_search
    Basic beam search example: https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

    Example
    -------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))
    token2id, first_batch, model = get_token2id_firstbatch_model()
    source, source_lengths = first_batch['feat'], first_batch['num_frames']
    sos_id, eos_id = int(token2id['<sos>']), int(token2id['<eos>']) # 2, 3
    max_dec_length = 4

    model.to(device)
    source, source_lengths = source.to(device), source_lengths.to(device)
    hypo, hypo_lengths, hypo_att, hypo_presoftmax = beam_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length, beam_size=2, expand_size=2, nbest=1)
    cropped_hypo, cropped_hypo_lengths, cropped_hypo_att, cropped_hypo_presoftmax = beam_search(model, source, source_lengths, sos_id, eos_id, max_dec_length, beam_size=2, expand_size=2, nbest=1)

    # # Each level of the search tree
    # ------time step 0------
    # [score: -0.026, tokenid_path:[5], state: [0], active: True],[score: -5.436, tokenid_path:[6], state: [0], active: True]
    # [score: -0.010, tokenid_path:[5], state: [1], active: True],[score: -7.122, tokenid_path:[6], state: [1], active: True]
    # [score: -0.044, tokenid_path:[5], state: [2], active: True],[score: -4.565, tokenid_path:[6], state: [2], active: True]
    # ------time step 1------
    # [score: -1.033, tokenid_path:[5, 4], state: [0, 0], active: True],[score: -1.087, tokenid_path:[5, 6], state: [0, 0], active: True]
    # [score: -1.117, tokenid_path:[5, 4], state: [1, 2], active: True],[score: -1.516, tokenid_path:[5, 7], state: [1, 2], active: True]
    # [score: -0.727, tokenid_path:[5, 6], state: [2, 4], active: True],[score: -1.198, tokenid_path:[5, 4], state: [2, 4], active: True]
    # ------time step 2------
    # [score: -1.316, tokenid_path:[5, 6, 3], state: [0, 0, 1], active: False],[score: -1.822, tokenid_path:[5, 4, 3], state: [0, 0, 0], active: False]
    # [score: -1.118, tokenid_path:[5, 4, 5], state: [1, 2, 2], active: True],[score: -2.382, tokenid_path:[5, 7, 4], state: [1, 2, 3], active: True]
    # [score: -0.962, tokenid_path:[5, 6, 3], state: [2, 4, 4], active: False],[score: -1.764, tokenid_path:[5, 4, 6], state: [2, 4, 5], active: True]
    # ------time step 3------
    # [score: -1.316, tokenid_path:[5, 6, 3], state: [0, 0, 1], active: False],[score: -1.822, tokenid_path:[5, 4, 3], state: [0, 0, 0], active: False]
    # [score: -1.823, tokenid_path:[5, 4, 5, 4], state: [1, 2, 2, 0], active: True],[score: -2.097, tokenid_path:[5, 4, 5, 7], state: [1, 2, 2, 0], active: True]
    # [score: -0.962, tokenid_path:[5, 6, 3], state: [2, 4, 4], active: False],[score: -1.764, tokenid_path:[5, 4, 6], state: [2, 4, 5], active: True]
    # ---hypo---
    # tensor([[5, 6, 3, 3],
    #         [5, 4, 5, 4],
    #         [5, 6, 3, 3]], device='cuda:0')
    # ---hypo_lengths---
    # tensor([ 3, -1,  3], device='cuda:0')
    # ---hypo_att---
    # tensor([[[0.0187, 0.9813],
    #          [0.0210, 0.9790],
    #          [0.0212, 0.9788],
    #          [0.0000, 0.0000]],

    #         [[0.0057, 0.9943],
    #          [0.0056, 0.9944],
    #          [0.0050, 0.9950],
    #          [0.0056, 0.9944]],

    #         [[1.0000, 0.0000],
    #          [1.0000, 0.0000],
    #          [1.0000, 0.0000],
    #          [0.0000, 0.0000]]], device='cuda:0', grad_fn=<CopySlices>)
    # ---hypo_presoftmax---
    # tensor([[[-2.0391e+00, -2.4686e+00, -3.3696e+00, -1.2013e+00, -1.9056e+00, 4.6328e+00,  1.2401e-01, -9.1314e-01],
    #          [-4.5584e+00, -5.5560e+00, -1.8747e+00,  2.1036e-03,  1.0197e+00, -3.7233e+00,  9.6586e-01,  2.8960e-02],
    #          [-2.8887e+00, -4.3951e+00, -2.3741e+00,  1.8658e+00, -2.3473e-01, -3.5218e+00,  2.5737e-01,  3.2377e-01],
    #          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,  0.0000e+00]],

    #         [[-1.0977e+00, -1.8111e+00, -3.2346e+00, -9.9084e-01, -2.3206e+00, 5.5821e+00, -3.4452e-01, -7.9397e-01],
    #          [-3.1162e+00, -4.4986e+00, -1.2099e+00, -6.0075e-02,  6.6851e-01, -2.0799e+00,  2.1094e-01,  2.7038e-01],
    #          [-1.5080e+00, -2.7002e+00, -2.3081e+00, -2.9946e-01, -1.3555e+00, 2.6545e+00, -4.2277e-01, -1.3397e-01],
    #          [-3.0643e+00, -4.4616e+00, -1.1970e+00, -2.8974e-02,  6.4926e-01, -2.0641e+00,  1.8507e-01,  2.8324e-01]],

    #         [[-2.2006e+00, -2.2896e+00, -3.6796e+00, -1.0538e+00, -1.8577e+00, 4.2987e+00,  5.3117e-01, -1.2819e+00],
    #          [-4.5086e+00, -4.8001e+00, -2.4802e+00, -1.3172e-01,  9.3378e-01, -3.6198e+00,  1.4054e+00, -6.8509e-01],
    #          [-2.6262e+00, -3.4670e+00, -2.7019e+00,  1.9906e+00, -3.1856e-01, -3.5389e+00,  6.1016e-01, -2.3925e-01],
    #          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,  0.0000e+00]]],
    #          device='cuda:0', grad_fn=<CopySlices>)

    # ---cropped_hypo---
    # [tensor([5, 6], device='cuda:0'),
    #  tensor([5, 4, 5, 4], device='cuda:0'),
    #  tensor([5, 6], device='cuda:0')]
    # ---cropped_hypo_lengths---
    # tensor([2, 4, 2], device='cuda:0')
    # ---cropped_hypo_att---
    # [tensor([[0.0187, 0.9813],
    #          [0.0210, 0.9790]], device='cuda:0', grad_fn=<SliceBackward>),
    #  tensor([[0.0057, 0.9943],
    #          [0.0056, 0.9944],
    #          [0.0050, 0.9950],
    #          [0.0056, 0.9944]], device='cuda:0', grad_fn=<AliasBackward>),
    #  tensor([[1.],
    #          [1.]], device='cuda:0', grad_fn=<SliceBackward>)]
    # ---cropped_hypo_presoftmax---
    # [tensor([[-2.0391e+00, -2.4686e+00, -3.3696e+00, -1.2013e+00, -1.9056e+00, 4.6328e+00,  1.2401e-01, -9.1314e-01],
    #          [-4.5584e+00, -5.5560e+00, -1.8747e+00,  2.1036e-03,  1.0197e+00, -3.7233e+00,  9.6586e-01,  2.8960e-02]],
    #          device='cuda:0', grad_fn=<SliceBackward>),
    #  tensor([[-1.0977, -1.8111, -3.2346, -0.9908, -2.3206,  5.5821, -0.3445, -0.7940],
    #          [-3.1162, -4.4986, -1.2099, -0.0601,  0.6685, -2.0799,  0.2109,  0.2704],
    #          [-1.5080, -2.7002, -2.3081, -0.2995, -1.3555,  2.6545, -0.4228, -0.1340],
    #          [-3.0643, -4.4616, -1.1970, -0.0290,  0.6493, -2.0641,  0.1851,  0.2832]],
    #          device='cuda:0', grad_fn=<SliceBackward>),
    #  tensor([[-2.2006, -2.2896, -3.6796, -1.0538, -1.8577,  4.2987,  0.5312, -1.2819],
    #          [-4.5086, -4.8001, -2.4802, -0.1317,  0.9338, -3.6198,  1.4054, -0.6851]],
    #          device='cuda:0', grad_fn=<SliceBackward>)]
    """
    batch_size = source.shape[0]
    # [batch_size*nbest, dec_length], [batch_size*best], [batch_size*nbest, dec_length, context_size] [batch_size*nbest, dec_length, dec_output_size]
    hypo, hypo_lengths, hypo_att, hypo_presoftmax = beam_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length,
                                                                      beam_size, expand_size, coeff_length_penalty, nbest)

    batch_size = len(source)
    index = []
    for i in range(batch_size): # [0, 1, 2] => [0, 0, 1, 1, 2, 2] if nbest = 2
        index += [i] * nbest
    context_lengths = mask2length(model.decoder.context_mask) # [batch_size]
    context_lengths = context_lengths.index_select(dim=0, index=source.new(index).long()) # [batch_size * nbest]; [3, 4, 2] => [3, 3, 4, 4, 2, 2]
    cropped_hypo_lengths = crop_hypothesis_lengths(hypo_lengths, max_dec_length) # remove eos_id
    cropped_hypo = [hypo[i][0:cropped_hypo_lengths[i]] for i in range(batch_size * nbest)]
    cropped_hypo_att = [hypo_att[i][0:cropped_hypo_lengths[i], 0:context_lengths[i]] for i in range(batch_size * nbest)]
    cropped_hypo_presoftmax = [hypo_presoftmax[i][0:cropped_hypo_lengths[i], :] for i in range(batch_size * nbest)]

    return cropped_hypo, cropped_hypo_lengths, cropped_hypo_att, cropped_hypo_presoftmax
