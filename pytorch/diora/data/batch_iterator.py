from diora.data.dataloader import FixedLengthBatchSampler, CharDataset, FlatDataset
from diora.blocks.negative_sampler import choose_negative_samples, NegativeSampler, calculate_freq_dist

import torch
from torch.utils.data import DataLoader
import numpy as np

from diora.data.segmenter import Word2i, Char2i
from diora.data.treeencoding import FlatEncoding, CharEncoding, CharBatchMap, BatchMap
from typing import Dict, Union, List, Optional, Tuple, TypedDict, Any, Literal, Generator, Iterator, Iterable

class IteratorOptions(TypedDict):
    cuda : bool
    ngpus : int
    num_workers : int
    local_rank : int
    k_neg : int
    batch_size : int
    length_to_size : Any    # TODO

class LossOptions(TypedDict):
    freq_dist_power : int

Config = Dict[str, Union[bool, int, None, DataLoader]]

def get_config(config : Config, **kwargs):
    for k, v in kwargs.items():
        if k in config:
            config[k] = v
    return config

def get_default_config() -> Config:
    return dict(
        batch_size=16,
        forever=False,
        drop_last=False,
        sort_by_length=True,
        shuffle=True,
        random_seed=None,
        filter_length=None,
        pin_memory=False,
        include_partial=False,
        cuda=False,
        ngpus=1,
        k_neg=3,
        negative_sampler=None,
        options_path=None,
        weights_path=None,
        vocab=None,
        length_to_size=None,
        local_rank=None,
    )


class Collate(object):
    @staticmethod
    def chunk(tensor, chunks, dim=0, i=0):
        if isinstance(tensor, torch.Tensor):
            return torch.chunk(tensor, chunks, dim=dim)[i]
        index = torch.chunk(torch.arange(len(tensor)), chunks, dim=dim)[i]
        return [tensor[ii] for ii in index]

    @staticmethod
    def partition(tensor, rank : int, device_ids : Iterable[int]):
        if tensor is None:
            return None
        if isinstance(tensor, dict):
            for k, v in tensor.items():
                tensor[k] = Collate.partition(v, rank, device_ids)
            return tensor
        return Collate.chunk(tensor, len(device_ids), 0, rank)

    def __init__(self, batch_iterator : "BatchIterator", rank, ngpus, char_collate : bool):
        self.batch_iterator : "BatchIterator" = batch_iterator
        self.rank = rank
        self.ngpus = ngpus

        self.char_collate = char_collate

    def collate_fn(self, batch) -> CharBatchMap:

        if self.char_collate:
            index, word_ids, char_ids = zip(*batch)
            #sents = torch.from_numpy(np.array(sents)).long()
            t_word_ids = torch.tensor(word_ids, dtype = torch.int)  # Change to long?
            t_char_ids = [[torch.tensor(word, dtype = torch.int) for word in sentence] for sentence in char_ids]

            if self.ngpus > 1:
                batch_map = CharBatchMap(input_ids = Collate.partition(t_word_ids, self.rank, range(self.ngpus)),
                                         char_ids = Collate.partition(t_char_ids, self.rank, range(self.ngpus)),
                                         length = t_word_ids.shape[1],
                                         batch_size =t_word_ids.shape[0],
                                         index = Collate.partition(index, self.rank, range(self.ngpus)))

            else:
                batch_map = CharBatchMap(input_ids = t_word_ids,
                                         char_ids = t_char_ids,
                                         length = t_word_ids.shape[1],
                                         batch_size =t_word_ids.shape[0],
                                         index = index)

            return batch_map

        else:
            index, word_ids = zip(*batch)
            #sents = torch.from_numpy(np.array(sents)).long()
            t_word_ids = torch.tensor(word_ids, dtype = torch.int)

            if self.ngpus > 1:
                batch_map = BatchMap(input_ids = Collate.partition(t_word_ids, self.rank, range(self.ngpus)),
                                         length = t_word_ids.shape[1],
                                         batch_size = t_word_ids.shape[0],
                                         index = Collate.partition(index, self.rank, range(self.ngpus)))
            else:
                batch_map = BatchMap(input_ids = t_word_ids,
                                     length = t_word_ids.shape[1],
                                     batch_size =t_word_ids.shape[0],
                                     index = index)

            return batch_map


class BatchIterator(object):

    def __init__(self, sentences : CharEncoding | FlatEncoding,
                 word2i : Word2i, iterator_options : IteratorOptions, **args):

        self.sentences : CharEncoding | FlatEncoding = sentences
        self.num_workers : int = iterator_options["num_workers"]
        print("Number of workers:", self.num_workers)
        self.config : Config = get_config(get_default_config(), **iterator_options, **args)
        self.loader : Optional[DataLoader] = None
        self.word2i : Word2i = word2i

    @property
    def dataset_size(self) -> int:
        return len(self.sentences["input_ids"])

    @property
    def dataset_minlen(self) -> int:
        return min(map(len, self.sentences["input_ids"]))

    @property
    def dataset_maxlen(self) -> int:
        return max(map(len, self.sentences["input_ids"]))

    @property
    def dataset_stats(self) -> str:
        return 'size={} minlen={} maxlen={}'.format(
            self.dataset_size, self.dataset_minlen, self.dataset_maxlen
        )

    def choose_negative_samples(self, negative_sampler, k_neg, device):
        return choose_negative_samples(negative_sampler, k_neg, device) #TODO

    def get_iterator(self, **kwargs : Config) -> Iterator[CharBatchMap] | Iterator[BatchMap]:
        config : Config = get_config(self.config.copy(), **kwargs)

        random_seed : int = config.get('random_seed')   # type: ignore
        batch_size : int = config.get('batch_size') # type: ignore
        filter_length : Optional[int] = config.get('filter_length') # type: ignore
        pin_memory : bool = config.get('pin_memory') # type: ignore
        include_partial : bool = config.get('include_partial') # type: ignore
        cuda : bool = config.get('cuda') # type: ignore
        n_gpus : int = config.get('ngpus')
        local_rank : int = config.get('local_rank') # type: ignore
        k_neg : int = config.get('k_neg') # type: ignore
        negative_sampler : None = config.get('negative_sampler', None) # type: ignore
        num_workers : int = self.num_workers
        length_to_size : int = config.get('length_to_size', None) # type: ignore

        #print("Iterator local_rank:", local_rank)
        #print("Iterator batch size:", batch_size)
        collate_fn = Collate(self, local_rank, n_gpus, char_collate = "char_ids" in self.sentences.keys()).collate_fn

        if self.loader is None:
            rng : np.random.RandomState = np.random.RandomState(seed = random_seed)

            dataset : FlatDataset | CharDataset
            if "char_ids" in self.sentences.keys():
                dataset = CharDataset(self.sentences)
            else:
                dataset = FlatDataset(self.sentences)

            sampler : FixedLengthBatchSampler
            sampler = FixedLengthBatchSampler(dataset,
                                              batch_size = batch_size,
                                              rng = rng,
                                              maxlen = filter_length,
                                              include_partial = include_partial,
                                              length_to_size = length_to_size)

            loader : DataLoader = DataLoader(dataset,
                                             shuffle = (sampler is None),
                                             num_workers = num_workers,
                                             pin_memory=pin_memory,
                                             batch_sampler=sampler,
                                             collate_fn=collate_fn)
            self.loader = loader

        def myiterator() -> Iterator[CharBatchMap] | Iterator[BatchMap]:   # TODO: implement random <unk> replacement

            batch : CharBatchMap | BatchMap
            assert self.loader is not None

            device = "cpu"
            if cuda:
                device = "cuda"
            if n_gpus > 1:
                device = local_rank
            
            for batch in self.loader:

                neg_samples = None
                if negative_sampler is not None:
                    neg_samples = self.choose_negative_samples(negative_sampler, k_neg, local_rank)

                sentences = batch["input_ids"]
                if "char_ids" in batch.keys():
                    chars = batch["char_ids"]

                if cuda:
                    sentences = sentences.to(device)
                    if "char_ids" in batch.keys():
                        chars = [[word.to(device) for word in sentence] for sentence in chars]

                if cuda and neg_samples is not None:
                    neg_samples["input_ids"] = neg_samples["input_ids"].to(device)
                    if "char_ids" in batch.keys():
                        neg_samples["char_ids"] = [[word.to(device) for word in sentence] for sentence in neg_samples["char_ids"]]

                if "char_ids" in batch.keys():
                    batch_map = CharBatchMap(input_ids = sentences,
                                             char_ids = chars,
                                             length = batch["length"],
                                             batch_size = sentences.shape[0],
                                             neg_samples = neg_samples,
                                             index = batch["index"])
                else:
                    batch_map = BatchMap(input_ids = sentences,
                                         length = batch["length"],
                                         batch_size = sentences.shape[0],
                                         neg_samples = neg_samples,
                                         index = batch["index"])

                yield batch_map

        return myiterator()


def make_batch_iterator(sentences : FlatEncoding | CharEncoding,
                        word2i : Word2i,
                        char2i : Optional[Char2i],
                        shuffle : bool,
                        include_partial : bool,
                        iterator_options : IteratorOptions,
                        loss_options : LossOptions,
                        filter_length : Optional[int] = None,
                        ) -> BatchIterator:

    vocab_size = len(word2i)

    input_ids = sentences["input_ids"]

    negative_sampler : Optional[NegativeSampler] = None
    
    #if len(input_ids) != 0:
    #    print("min", min([int(min(s)) for s in input_ids if len(s) != 0]), "max", max([int(max(s)) for s in input_ids if len(s) != 0]))
    freq_dist = calculate_freq_dist(input_ids, vocab_size)
    negative_sampler  = NegativeSampler(freq_dist = freq_dist, dist_power = loss_options["freq_dist_power"],
                                        word2idx = word2i, char2idx = char2i)

    vocab_lst = [w for w, _ in sorted(word2i.items(), key=lambda x: x[1])]

    batch_iterator = BatchIterator(
        sentences,
        word2i,
        iterator_options = iterator_options,
        shuffle = shuffle,
        include_partial = include_partial,
        filter_length=filter_length,
        negative_sampler=negative_sampler,
        vocab=vocab_lst
        )

    return batch_iterator


