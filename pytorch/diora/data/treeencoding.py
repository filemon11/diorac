from transformers import BatchEncoding
import torch

from diora.data.representations import Tree

from typing import TypedDict, List, Union
from typing_extensions import NotRequired

class FlatEncoding(TypedDict):
    input_ids : Union[List[int], List[List[int]]]
    attention_mask : Union[List[int], List[List[int]]]
    tokens : Union[List[str], List[List[str]]]


class TreeEncoding(FlatEncoding):
    trees : Union[Tree, List[Tree]]


class CharEncoding(FlatEncoding):
    char_ids : Union[List[List[int]], List[List[List[int]]]]


class CharTreeEncoding(TreeEncoding):
    char_ids : Union[List[List[int]], List[List[List[int]]]]


class _CoreBatchMap(TypedDict):
    input_ids : torch.Tensor
    length : int
    batch_size : int
    index : List[int]
    trees : NotRequired[List[Tree]]
    tokens : NotRequired[List[List[str]]]


class BatchMap(_CoreBatchMap):
    neg_samples : NotRequired["BatchMap"]


class CharBatchMap(BatchMap):
    char_ids : List[List[torch.Tensor]]


class NegExampleBatchMap(_CoreBatchMap):
    neg_samples : BatchMap


class NegExampleCharBatchMap(NegExampleBatchMap):
    char_ids : List[List[torch.Tensor]]