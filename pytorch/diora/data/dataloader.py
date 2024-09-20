import os

import torch
from torch.utils.data import Sampler

import numpy as np

from diora.logging.configuration import get_logger, Logger
from diora.data.treeencoding import CharEncoding, FlatEncoding

from typing import List, Tuple, Optional, Dict, TypedDict, Iterator

class State(TypedDict):
    nbatches : int
    surplus : bool
    position : int

class FixedLengthBatchSampler(Sampler):

    def __init__(self, data_source : "FlatDataset | CharDataset", 
                 batch_size : int, 
                 include_partial : bool = False, 
                 rng : Optional[np.random.RandomState] = None, 
                 maxlen : Optional[int] = None,
                 length_to_size : Optional[Dict[int, int]] = None): # TODO
        
        self.data_source : FlatDataset | CharDataset = data_source
        self.active : bool= False

        self.rng : np.random.RandomState
        self.rng = np.random.RandomState(seed=11) if rng is None else rng
        
        self.batch_size : int = batch_size
        self.maxlen : Optional[int] = maxlen
        self.include_partial : bool = include_partial
        self.length_to_size : Optional[Dict[int, int]] = length_to_size
        self._batch_size_cache : Dict[int, int] = {0 : self.batch_size}
        self.logger : Logger = get_logger()

    def get_batch_size(self, length : int) -> int:
        if self.length_to_size is None:
            return self.batch_size
        
        if length in self._batch_size_cache:
            return self._batch_size_cache[length]
        
        start : int = max(self._batch_size_cache.keys())
        batch_size : int = self._batch_size_cache[start]

        for n in range(start+1, length+1):
            if n in self.length_to_size:
                batch_size = self.length_to_size[n]

            self._batch_size_cache[n] = batch_size

        return batch_size

    def reset(self) -> None:
        """
        Create a map of {length: List[example_id]} and maintain how much of
        each list has been seen.

        If include_partial is False, then do not provide batches that are below
        the batch_size.

        If length_to_size is set, then batch size is determined by length.

        """

        # Record the lengths of each example.
        length_map : Dict[int, List[int]] = {}

        x : List[int]
        length : int
        for i in range(len(self.data_source)):
            x = self.data_source.dataset["input_ids"][i]
            length = len(x)

            if (self.maxlen is not None 
                    and self.maxlen > 0 
                    and length > self.maxlen):
                continue

            length_map.setdefault(length, []).append(i)

        # Shuffle the order.
        for length in length_map.keys():
            self.rng.shuffle(length_map[length])

        # Initialize state.
        state : Dict[int, State] = {}

        arr : List[int]
        batch_size : int
        nbatches : int
        surplus : bool
        for length, arr in length_map.items():
            batch_size = self.get_batch_size(length)
            nbatches = len(arr) // batch_size
            surplus = nbatches * batch_size < len(arr)

            state[length] = State(nbatches = nbatches, 
                                  surplus = surplus, 
                                  position = -1)

        # Batch order, in terms of length.
        order : List[int] = []

        v : State
        for length, v in state.items():
            order += [length] * v['nbatches']

        ## Optionally, add partial batches.
        if self.include_partial:
            for length, v in state.items():
                if v['surplus']:
                    order += [length]

        self.rng.shuffle(order)

        self.length_map = length_map
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self) -> List[int]:
        index : int = self.index + 1

        length : int = self.order[index]
        batch_size : int = self.get_batch_size(length)
        #print("get next batch batch size:", batch_size)
        position : int = self.state[length]['position'] + 1
        start : int = position * batch_size
        batch_index : List[int] = self.length_map[length][start:start+batch_size]

        self.state[length]['position'] = position
        self.index = index

        return batch_index

    def __iter__(self) -> Iterator[List[int]]:
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self) -> int:
        return len(self.order)
    

class FlatDataset(torch.utils.data.Dataset):

    def __init__(self, dataset : FlatEncoding):
        self.dataset : FlatEncoding = dataset

    def __getitem__(self, index : int) -> Tuple[int, List[int]]:

        word_ids : List[int] = self.dataset["input_ids"][index]
        return index, word_ids

    def __len__(self) -> int:
        return len(self.dataset["input_ids"])
    

class CharDataset(torch.utils.data.Dataset):

    def __init__(self, dataset : CharEncoding):
        self.dataset : CharEncoding = dataset

    def __getitem__(self, index : int) -> Tuple[int, List[int], List[List[int]]]:
        
        word_ids : List[int] = self.dataset["input_ids"][index]
        char_ids : List[List[int]] = self.dataset["char_ids"][index]
        return index, word_ids, char_ids

    def __len__(self) -> int:
        return len(self.dataset["input_ids"])
