"""Unified interface for training and predicting with
unlabelled parsers. TODO: use it"""

import torch.nn as nn
import torch

from diora.data.representations import Tree
from diora.data.treeencoding import BatchMap, CharBatchMap

from typing import Protocol, Generic, TypedDict, TypeVar, Any, Dict, Literal, Optional, List, Tuple, Union, Type
from typing_extensions import Self, NotRequired

from abc import ABC, abstractmethod, abstractproperty

DioraName = Literal["treelstm", "mlp", "mlp-shared"]


class TrainerOptions(TypedDict):
    emb_dim : int
    lr : float
    cuda : bool
    multigpu : bool
    ngpus : int
    load_model_path : Optional[str]
    default_idx : int
    experiment_name : str
    embeddings_grad : bool
    

class DioraTrainerOptions(TrainerOptions):
    hidden_dim : int
    k_neg : int
    normalize : bool
    local_rank : int
    master_addr : str
    master_port : str
    arch : DioraName
    reconstruct_mode : Literal["margin", "softmax"]
    margin : int


class StructFormerTrainerOptions(TrainerOptions):
    pass    # TODO




O = TypeVar("O", bound = TrainerOptions)
BM_T = TypeVar("BM_T", bound = BatchMap)
BM_P = TypeVar("BM_P", bound = BatchMap)
M = TypeVar("M", bound = nn.Module)


class Embedding(ABC, nn.Module, Generic[BM_P]):
    @abstractmethod
    def forward(self, batch : BM_P) -> torch.Tensor:
        ...


class Trainer(ABC, Generic[O, BM_T, BM_P]):
    """Defines an interface for unsupervised parsing trainers
    and predictors."""
    @abstractmethod
    def __init__(self, options : O, embeddings : Embedding[BM_P]) -> None:
        ...

    @abstractmethod
    def train_step(self, batch_map : BM_T) -> Dict[str, int | float]:
        ...

    @abstractmethod
    def pred_step(self, batch_map : BM_P) -> Tree: 
        ...
        # extends beyond ListTrees to allow for models that also deduce
        # node labels; unlabelled trees simply receive dummy labels

    @abstractmethod
    def save_model(self, path : str) -> None:
        ...

    @abstractmethod
    def load_model(cls, path : str) -> None:
        ...

class NeuralTrainer(Trainer[O, BM_T, BM_P], Generic[O, BM_T, BM_P, M]):

    @abstractproperty
    def net(self) -> M:
        ...

    @abstractproperty
    def optimizer(self) -> torch.optim.Optimizer:
        ...
    
    @optimizer.setter
    @abstractmethod
    def optimizer(self) -> None:
        ...
        
    @abstractproperty
    def optimizer_cls(self) -> Type[torch.optim.Optimizer]:
        ...

    @abstractproperty
    def optimizer_kwargs(self) -> Dict[str, Any]:
        ...

    def init_optimizer(self, optimizer_cls : Optional[Type[torch.optim.Optimizer]],
                    optimizer_kwargs : Optional[Dict[str, Any]]) -> None:
        if optimizer_cls is None:
            optimizer_cls = self.optimizer_cls

        if optimizer_kwargs is None:
            optimizer_kwargs = self.optimizer_kwargs

        params : List[nn.Parameter] = [p for p in self.net.parameters() if p.requires_grad]

        self.optimizer = optimizer_cls(params, **optimizer_kwargs)

    def save_model(self, model_file : str) -> None:
        state_dict : Dict[str, Any] = self.net.state_dict()

        todelete : List[str] = []

        for k in state_dict.keys():
            if 'embeddings' in k:
                todelete.append(k)

        for k in todelete:
            del state_dict[k]

        torch.save({'state_dict': state_dict}, model_file)

    def load_model(self, model_file : str) -> None:
        save_dict = torch.load(model_file, map_location = lambda storage, loc: storage)

        state_dict_toload : Dict[str, Any] = save_dict['state_dict']

        state_dict_net : Dict[str, Any] = self.net.state_dict()

        keys : List[str]

        # Bug related to multi-gpu
        keys = list(state_dict_toload.keys())
        prefix : str = 'module.'

        newk : str
        for k in keys:
            if k.startswith(prefix):
                newk = k[len(prefix):]
                state_dict_toload[newk] = state_dict_toload[k]
                del state_dict_toload[k]

        # Remove extra keys.
        keys = list(state_dict_toload.keys())
        for k in keys:
            if k not in state_dict_net:
                print('deleting {}'.format(k))
                del state_dict_toload[k]

        # Hack to support embeddings.
        for k in state_dict_net.keys():
            if 'embeddings' in k:
                state_dict_toload[k] = state_dict_net[k]

        self.net.load_state_dict(state_dict_toload)

    def gradient_update(self, loss : torch.Tensor) -> None:
        assert self.optimizer is not None, "Trainer is missing an optimizer."

        self.optimizer.zero_grad()

        loss.backward()
        
        params : List[nn.Parameter] = [p for p in self.net_params if p.requires_grad]

        torch.nn.utils.clip_grad_norm_(params, 5.0)

        self.optimizer.step()

    @abstractproperty
    def net_params(self) -> List[nn.Parameter]:
        ...

    @abstractproperty
    def named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        ...

class MLMTrainer(NeuralTrainer[O, BM_T, BM_P, M], ABC):
    @abstractmethod
    def unary_probablities(self, batch_map : BM_P) -> torch.Tensor:
        # Input: B x L, Output: B x L x V
        ...

class LMTrainer(NeuralTrainer[O, BM_T, BM_P, M], ABC):
    @abstractmethod
    def next_probabilities(self, batch_map : BM_P) -> torch.Tensor:
        # Input: B x L, Output: B x V
        ...
    @abstractmethod
    def sentence_probabilities(self, batch_map : BM_P) -> torch.Tensor:
        # Input: B x L, Output: B x 1
        ...