import os
import sys
import traceback
import types

import torch
import torch.nn as nn
import torch.optim as optim

from diora.net.diora import DioraTreeLSTM, DioraMLP, DioraMLPShared

from diora.analysis.cky import ParsePredictor as CKY

from diora.data.batch_iterator import BatchIterator
from diora.data.representations import Tree
from diora.net.parser_types import NeuralTrainer, DioraTrainerOptions, MLMTrainer, LMTrainer
from diora.data.treeencoding import NegExampleBatchMap, NegExampleCharBatchMap, BatchMap, CharBatchMap
from diora.net.custom_embeddings import DualEmbed, WordEmbedding, CharEmbedding, DualTransformerEmbed, SimpleEmbedding
from diora.data.tokenizer import Word2i
from diora.data.segmenter import Char2i

from diora.logging.configuration import get_logger

from typing import Optional, List, Tuple, Dict, Literal, TypedDict, Union, Any, Type, TypeVar
from typing_extensions import NotRequired

TupleTree = "Literal[0] | Tuple[TupleTree | int, ...]"

LossFunc = Union["ReconstructionLoss", "ReconstructionSoftmaxLoss"]
LossName = Literal["reconstruction_softmax_loss", "reconstruction_loss"]

Diora = Union[DioraTreeLSTM, DioraMLP, DioraMLPShared]

class RecLossCombinedResult(TypedDict, total = False):
    reconstruction_softmax_loss : torch.Tensor
    reconstruction_loss : torch.Tensor

class LossDict(RecLossCombinedResult):
    total_loss : torch.Tensor

class ModelOutput(TypedDict):
    reconstruction_softmax_loss : NotRequired[float]
    reconstruction_loss : NotRequired[float]
    total_loss : float
    batch_size : int
    length : int

class Info(TypedDict):
    pass

class ReconstructionLoss(nn.Module):
    name : str = 'reconstruct_loss'

    def __init__(self, embeddings : WordEmbedding | CharEmbedding,
                 input_size : int,
                 size : int,
                 margin : int = 1,
                 k_neg : int = 3,
                 cuda : int | Literal[False] = False):

        super(ReconstructionLoss, self).__init__()

        self.k_neg : int = k_neg
        self.margin : int = margin

        self.embeddings : WordEmbedding | CharEmbedding = embeddings

        self.mat : nn.Parameter
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))

        self._cuda : int | Literal[False] = cuda

        self.reset_parameters()

    def reset_parameters(self):
        params : List[nn.Parameter] = [p for p in self.parameters() if p.requires_grad]

        for param in params:
            param.data.normal_()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, sentences : NegExampleBatchMap | NegExampleCharBatchMap,
                diora : Diora) -> Tuple[torch.Tensor, str, torch.Tensor]:

        batch_size : int = sentences["batch_size"]
        length : int = sentences["length"]

        k : int = self.k_neg

        emb_pos : torch.Tensor = self.embeddings(sentences)
        emb_neg : torch.Tensor = self.embeddings(sentences["neg_samples"]).squeeze(0)

        # Calculate scores.

        ## The predicted vector.
        cell = diora.outside_h[:, :length].view(batch_size, length, 1, -1)

        ## The projected samples.
        proj_pos : torch.Tensor = torch.matmul(emb_pos, torch.t(self.mat))
        proj_neg : torch.Tensor = torch.matmul(emb_neg, torch.t(self.mat))

        ## The score.
        xp : torch.Tensor = torch.einsum('abc,abxc->abx', proj_pos, cell)
        xn : torch.Tensor = torch.einsum('ec,abxc->abe', proj_neg, cell)
        score : torch.Tensor = torch.cat([xp, xn], 2)

        # Calculate loss.
        lossfn : nn.MultiMarginLoss = nn.MultiMarginLoss(margin=self.margin)

        indices : torch.Tensor = torch.arange(1, length + 1).repeat(batch_size)
        # Remove last element prediction (future tag)
        inputs : torch.Tensor = score.view(batch_size * length, k + 1)  #[(indices != 1) & (indices != length)]

        #device : Optional[int] = torch.cuda.current_device() if self._cuda else None
        device = self.device

        outputs : torch.Tensor = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device = device)

        loss : torch.Tensor = lossfn(inputs, outputs)

        return loss, "reconstruction_loss", loss

    def forward_probs(self, sentences : BatchMap | CharBatchMap,
                     all_indices : BatchMap | CharBatchMap,
                     diora : Diora) -> torch.Tensor:

        batch_size : int = sentences["batch_size"]
        length : int = sentences["length"]

        emb_all : torch.Tensor = self.embeddings(all_indices)

        ## The predicted vector.
        cell = diora.outside_h[:, :length].view(batch_size, length, 1, -1)

        ## The projected samples
        proj_all : torch.Tensor = torch.matmul(emb_all, torch.t(self.mat))

        ## The score.
        xn : torch.Tensor = torch.einsum('ec,abxc->abe', proj_all, cell)[:, :-1, :]

        return xn


class ReconstructionSoftmaxLoss(nn.Module):
    name : str = 'reconstruct_softmax_loss'

    def __init__(self, embeddings : WordEmbedding | CharEmbedding,
                 input_size : int,
                 size : int,
                 margin : int = 1,
                 k_neg : int = 3,
                 cuda : int | Literal[False] = False):

        super(ReconstructionSoftmaxLoss, self).__init__()
        self.k_neg : int = k_neg
        self.margin : int = margin
        self.input_size : int = input_size

        self.embeddings : WordEmbedding | CharEmbedding = embeddings
        self.mat : nn.Parameter = nn.Parameter(torch.FloatTensor(size, input_size))
        self._cuda : int | Literal[False] = cuda

        self.reset_parameters()

    def reset_parameters(self):
        params : List[nn.Parameter] = [p for p in self.parameters() if p.requires_grad]
        for param in params:
            param.data.normal_()

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, sentences : NegExampleBatchMap | NegExampleCharBatchMap,
                diora : Diora) -> Tuple[torch.Tensor, str, torch.Tensor]:

        batch_size : int = sentences["batch_size"]
        length : int = sentences["length"]

        k : int = self.k_neg

        emb_pos : torch.Tensor = self.embeddings(sentences)
        emb_neg : torch.Tensor = self.embeddings(sentences["neg_samples"])#.squeeze(0)

        # Calculate scores.

        ## The predicted vector.
        #print("Outside device:", diora.outside_h.device)
        #print(diora.outside_h.shape, batch_size, length, )
        cell : torch.Tensor = diora.outside_h[:, :length].reshape(batch_size, length, 1, -1)
        #cell : torch.Tensor = diora.outside_h[:, :length].view(batch_size, length, 1, -1)

        ## The projected samples.
        proj_pos : torch.Tensor = torch.matmul(emb_pos, torch.t(self.mat))
        proj_neg : torch.Tensor = torch.matmul(emb_neg, torch.t(self.mat))

        #print(proj_pos.shape, cell.shape)
        ## The score.
        xp : torch.Tensor = torch.einsum('abc,abxc->abx', proj_pos, cell)
        xn : torch.Tensor = torch.einsum('zec,abxc->abe', proj_neg, cell)
        #score = xn
        score : torch.Tensor = torch.cat([xp, xn], 2) #[:, :-1]        # TODO

        # Calculate loss.
        lossfn : nn.CrossEntropyLoss = nn.CrossEntropyLoss()

        indices : torch.Tensor = torch.arange(1, length + 1).repeat(batch_size)
        # Remove last element prediction (future tag)
        inputs : torch.Tensor = score.view(batch_size * length, k + 1)  #[(indices != 1) & (indices != length)]

        #device : Optional[int] = torch.cuda.current_device() if self._cuda else None
        device = self.device

        #outputs = sentences["input_ids"].view(batch_size * length)[(indices != 1) & (indices != length)].to(torch.int64)
        outputs : torch.Tensor = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device = device)

        loss : torch.Tensor = lossfn(inputs, outputs)

        return loss, "reconstruction_softmax_loss", loss

    def forward_probs(self, sentences : BatchMap | CharBatchMap,
                     all_indices : BatchMap | CharBatchMap,
                     diora : Diora) -> torch.Tensor:

        batch_size : int = sentences["batch_size"]
        length : int = sentences["length"]

        emb_all : torch.Tensor = self.embeddings(all_indices)

        ## The predicted vector.
        cell = diora.outside_h[:, :length].view(batch_size, length, 1, -1)
        cell = nn.functional.normalize(cell, dim = 3)

        ## The projected samples
        proj_all : torch.Tensor = torch.matmul(emb_all, torch.t(self.mat))
        proj_all = nn.functional.normalize(proj_all, dim = 1)

        ## The score.
        xn : torch.Tensor = torch.einsum('ec,abxc->abe', proj_all, cell)[:, :-1, :]

        return nn.Softmax(dim=2)(xn)


def get_loss_funcs(options : "BuildNetOptions",
                   embedding_layer : WordEmbedding | CharEmbedding,
                   input_dim : int) -> List[LossFunc]:

    size : int = options["hidden_dim"]
    k_neg : int = options["k_neg"]
    margin : int = options["margin"]
    cuda : Literal[False] | int = options["cuda"]

    loss_funcs : List[LossFunc] = []

    reconstruction_loss_fn : LossFunc

    # Reconstruction Loss
    if options["reconstruct_mode"] == 'margin':
        reconstruction_loss_fn = ReconstructionLoss(embedding_layer,
            margin=margin, k_neg=k_neg, input_size=input_dim, size=size, cuda=cuda)

    elif options["reconstruct_mode"] == 'softmax':
        reconstruction_loss_fn = ReconstructionSoftmaxLoss(embedding_layer,
            margin=margin, k_neg=k_neg, input_size=input_dim, size=size, cuda=cuda)

    loss_funcs.append(reconstruction_loss_fn)

    return loss_funcs


class Net(nn.Module):
    def __init__(self, embed : WordEmbedding | CharEmbedding, diora, loss_funcs : List[LossFunc] = [],
                 unk_index : int = 3,
                 replace_freq : float = 0.0):
        super(Net, self).__init__()

        self.embed : WordEmbedding | CharEmbedding = embed
        self.diora : Diora = diora

        self.unk_index : int = unk_index
        """Index to insert for random word replacement."""

        self.replace_freq : float = replace_freq
        """Frequency of random word replacement in [0, 1]"""

        self.loss_funcs : nn.ModuleDict = nn.ModuleDict({m.name : m for m in loss_funcs})

        self.reset_parameters()

    def reset_parameters(self):
        params : List[nn.Parameter] = [p for p in self.parameters() if p.requires_grad]
        for param in params:
            param.data.normal_()

    def compute_loss(self, batch : NegExampleBatchMap | NegExampleCharBatchMap) \
                                        -> Tuple[RecLossCombinedResult, torch.Tensor]:

        ret : RecLossCombinedResult = {}
        loss : List[torch.Tensor] = []

        # Loss
        diora : Diora = self.diora.get_chart_wrapper()

        subloss : torch.Tensor
        loss_name : LossName
        value : torch.Tensor

        for func in self.loss_funcs.values():
            subloss, loss_name, value = func(batch, diora)       #HERE

            loss.append(subloss.view(1, 1))

            ret[loss_name] = value

        loss_tensor : torch.Tensor = torch.cat(loss, 1)

        return ret, loss_tensor

    def forward(self, batch : CharBatchMap | BatchMap,
                compute_loss : bool = True) -> LossDict:

        # Embed
        #print("Batch device:", batch["input_ids"].device)
        #print("Batch size:", batch["batch_size"], "Tensor size:", batch["input_ids"].shape)
        embed : torch.Tensor = self.embed(batch)
        #print("Embedded batch device:", embed.device)
        #print("Embedding size:", embed.shape)

        # Run DIORA
        self.diora(embed)

        ret : RecLossCombinedResult
        loss : torch.Tensor
        # Compute Loss
        if compute_loss:
            assert "neg_samples" in batch.keys(), "neg_samples cannot be None if compute_loss == True"

            ret, loss = self.compute_loss(batch)

        else:
            ret, loss = {}, torch.full((1, 1), 1, dtype=torch.float32,
                                       device=embed.device)

        # Results
        out_ret : LossDict = dict(total_loss = loss)
        out_ret.update(ret)

        return out_ret

    def forward_probs(self, batch : CharBatchMap, all_indices : torch.Tensor) -> torch.Tensor:
        # Embed
        embed : torch.Tensor = self.embed(batch)

        # Run DIORA
        self.diora(embed)

        # Compute Loss
        output : torch.Tensor = self.loss_funcs["reconstruct_softmax_loss"].forward_probs(batch, all_indices, self.diora)

        return output

T = TypeVar("T", bound = nn.Module)


class DioraTrainer(NeuralTrainer[DioraTrainerOptions, NegExampleBatchMap, BatchMap, Net]):
    def __init__(self, options : DioraTrainerOptions, embeddings : WordEmbedding | CharEmbedding) -> None:
        self.embeddings : WordEmbedding | CharEmbedding = embeddings
        """Embedding module that receives an integer tensor
        and returns a float tensor.
        B x L => B x L x E with
        - B: batch size
        - L: sequence length
        - E: embedding dim (`options.emb_dim`)"""

        logger = get_logger()

        lr : float = options["lr"]
        size : int = options["hidden_dim"]
        normalize : bool = options["normalize"]

        self.cuda : bool = options["cuda"]
        self.rank : int = options["local_rank"]

        # Diora
        name_to_diora : Dict[DioraName, Type[Diora]] = {'treelstm' : DioraTreeLSTM,
                                                        'mlp' : DioraMLP,
                                                        'mlp-shared' : DioraMLPShared}

        self.diora : Diora = name_to_diora[options["arch"]](size,
                                                       outside=True,
                                                       normalize=normalize,
                                                       compress=False)

        # Loss
        loss_funcs : List[LossFunc] = get_loss_funcs(options, embeddings, size)

        # Net
        self.net : Net = Net(embeddings, self.diora, loss_funcs = loss_funcs, unk_index = options["unk_idx"])

        # Load model.
        if options["load_model_path"] is not None:
            logger.info('Loading model: {}'.format(options["load_model_path"]))
            self.load_model(options["load_model_path"])

        # CUDA-support
        if self.cuda:
            self.net.cuda()
            self.diora.cuda()

        self.optimizer : Optional[optim.Optimizer] = None
        self.optimizer_cls : Optional[Type[optim.Optimizer]] = None
        self.optimizer_kwargs : Optional[Dict[str, Any]] = None

        self.experiment_name : Optional[str] = options["experiment_name"]

        self.init_optimizer(optim.Adam, dict(lr=lr, betas=(0.9, 0.999), eps=1e-8))

        self.parse_predictor = CKY()

    def train_step(self, batch_map: BatchMap) -> ModelOutput:
        return self.step(batch_map,
                         train = True,
                         compute_loss = True)

    def prepare_pred(self) -> None:
        def override_init_with_batch(var):
            init_with_batch = var.init_with_batch

            def func(self, *args, **kwargs):
                init_with_batch(*args, **kwargs)
                self.saved_scalars = {i: {} for i in range(self.length)}
                self.saved_scalars_out = {i: {} for i in range(self.length)}

            var.init_with_batch = types.MethodType(func, var)


        def override_inside_hook(var):
            def func(self, level, h, c, s):
                length = self.length
                B = self.batch_size
                L = length - level

                assert s.shape[0] == B
                assert s.shape[1] == L
                # assert s.shape[2] == N
                assert s.shape[3] == 1
                assert len(s.shape) == 4
                smax = s.max(2, keepdim=True)[0]
                s = s - smax

                for pos in range(L):
                    self.saved_scalars[level][pos] = s[:, pos, :]

            var.inside_hook = types.MethodType(func, var)

        ## Monkey patch parsing specific methods.
        override_init_with_batch(self.diora)
        override_inside_hook(self.diora)

    def pred_step(self, batch_map : BatchMap) -> Tree:

        ## Turn off outside pass.
        self.diora.outside = False

        ## Eval mode.
        self.net.eval()

        batch_size = batch_map["batch_size"]
        length = batch_map["length"]

        # Skip very short sentences.
        trees : List[TupleTree]
        if length <= 2:
            if length == 2:
                trees = [(0, 1) for _ in range(batch_size)]
            elif length == 1:
                trees = [(0, ) for _ in range(batch_size)]
            else:
                raise ValueError

            return trees

        _ = self.step(batch_map, train=False, compute_loss=False)

        trees = self.parse_predictor.parse_batch(batch_map, self.net.saved_scalars, self.net.device)

        # => turn into tree with word indices as leaves
        def init_tree(node : TupleTree | int, leaves : List[int]) -> Tree:
            if isinstance(node, int):
                return Tree("DT", str(leaves.pop()))

            else:
                children : List[Tree] = []
                for child in node:
                    children.append(init_tree(child, leaves))
                return Tree("S", children)

        return [init_tree(t, list(reversed(s.tolist()))) for t, s in zip(trees, batch_map["sentences"])]

    def step(self, batch_map : BatchMap, train : bool = True, compute_loss : bool = True) -> ModelOutput:

        if train:
            self.net.train()
        else:
            self.net.eval()

        with torch.set_grad_enabled(train):
            model_output : LossDict = self.run_net(batch_map, compute_loss = compute_loss)

        # Calculate average loss for multi-gpu and sum for backprop.
        total_loss : torch.Tensor = model_output['total_loss'].mean(dim = 0).sum()

        if train:
            self.gradient_update(total_loss)

        result : ModelOutput = self.prepare_result(batch_map, model_output)

        return result

    def run_net(self, batch_map : BatchMap,
                compute_loss : bool = True) -> LossDict:

        out : LossDict = self.net(batch_map,
                                  compute_loss = compute_loss)

        return out

    def prepare_result(self, batch_map : BatchMap, model_output : LossDict) -> ModelOutput:
        result : ModelOutput = dict(batch_size = batch_map['batch_size'],
                                    length = batch_map['length'],
                                    **{k : v.mean(dim = 0).sum().item() # type: ignore
                                            for k, v in model_output.items() if 'loss' in k})

        return result

class HyMLMDioraTrainer(DioraTrainer, MLMTrainer, LMTrainer):
    ...

class Trainer():
    def __init__(self, net : Net, k_neg = None, ngpus : Optional[int] = None,
                 cuda : Optional[int] = None, rank : Optional[int] = None, experiment_name : Optional[str] = None):
        self.net : Net = net
        self.optimizer : Optional[optim.Optimizer] = None
        self.optimizer_cls : Optional[Type[optim.Optimizer]] = None
        self.optimizer_kwargs : Optional[Dict[str, Any]] = None
        self.cuda : Optional[int] = cuda
        self.ngpus : Optional[int] = ngpus
        self.rank : Optional[int] = rank
        self.experiment_name : Optional[str] = experiment_name

        self.parallel_model = None

        print("Trainer initialized with {} gpus.".format(ngpus))

    def freeze_diora(self):
        p : nn.Parameter
        for p in self.net.diora.parameters():
            p.requires_grad = False

    def parameter_norm(self, requires_grad : bool = True, diora = False) -> float:

        net = self.net.diora if diora else self.net

        total_norm : float = 0
        for p in net.parameters():
            if requires_grad and not p.requires_grad:
                continue

            total_norm += p.norm().item()

        return total_norm

    def init_optimizer(self, optimizer_cls : Type[optim.Optimizer], optimizer_kwargs : Dict[str, Any]) -> None:
        if optimizer_cls is None:
            optimizer_cls = self.optimizer_cls

        if optimizer_kwargs is None:
            optimizer_kwargs = self.optimizer_kwargs

        params : List[nn.Parameter] = [p for p in self.net.parameters() if p.requires_grad]

        self.optimizer = optimizer_cls(params, **optimizer_kwargs)

    @staticmethod
    def get_single_net(net : T) -> T:
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            return net.module
        return net

    def save_model(self, model_file : str) -> None:
        state_dict : Dict[str, Any] = self.net.state_dict()

        todelete : List[str] = []

        for k in state_dict.keys():
            if 'embeddings' in k:
                todelete.append(k)

        for k in todelete:
            del state_dict[k]

        torch.save({'state_dict': state_dict}, model_file)

    @staticmethod
    def load_model(net : T, model_file : str):
        save_dict = torch.load(model_file, map_location = lambda storage, loc: storage)
        state_dict_toload : Dict[str, Any] = save_dict['state_dict']
        state_dict_net : Dict[str, Any] = Trainer.get_single_net(net).state_dict()

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

        Trainer.get_single_net(net).load_state_dict(state_dict_toload)

    def run_net(self, batch_map : BatchMap,
                compute_loss : bool = True) -> Dict[str, torch.Tensor]:

        #for i in self.net.named_parameters():
        #    print(f"{i[0]} -> {i[1].device}")
        out : Dict[str, torch.Tensor] = self.net(batch_map,
                                                 compute_loss = compute_loss)

        return out

    def gradient_update(self, loss : torch.Tensor, do_step : bool = True) -> None:
        assert self.optimizer is not None, "Trainer is missing an optimizer."
        #self.optimizer.zero_grad()

        loss.backward()

        params : List[nn.Parameter] = [p for p in self.net.parameters() if p.requires_grad]

        if do_step:
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def prepare_result(self, batch_map : BatchMap, model_output : Dict[str, torch.Tensor]) -> Dict[str, float | int]:
        result : Dict[str, float | int] = {}

        result['batch_size'] = batch_map['batch_size']
        result['length'] = batch_map['length']

        for k, v in model_output.items():
            if 'loss' in k:
                result[k] = v.mean(dim=0).sum().item()

        return result

    def prepare_info(self) -> Info:
        return {}

    def step(self, *args, **kwargs):
        try:
            return self._step(*args, **kwargs)

        except Exception as err:
            batch_map : BatchMap = args[0]
            print('Failed with shape: {}'.format(batch_map['input_ids'].shape))

            if self.ngpus > 1:
                print(traceback.format_exc())
                print('The step failed. Running multigpu cleanup.')
                os.system("ps -elf | grep [p]ython | grep adrozdov | grep " + self.experiment_name + " | tr -s ' ' | cut -f 4 -d ' ' | xargs -I {} kill -9 {}")
                sys.exit()

            else:
                raise err

    def _step(self, batch_map : BatchMap, train : bool = True, compute_loss : bool = True, do_step : bool = True) -> Dict[str, int | float]:
        if train:
            self.net.train()
        else:
            self.net.eval()

        assert self.ngpus is not None, "Trainer.ngpus cannot be None."

        with torch.set_grad_enabled(train):
            model_output : Dict[str, torch.Tensor] = self.run_net(batch_map, compute_loss=compute_loss)

        # Calculate average loss for multi-gpu and sum for backprop.
        total_loss : torch.Tensor = model_output['total_loss'].mean(dim=0).sum()

        if train:
            self.gradient_update(total_loss, do_step = do_step)

        result : Dict[str, int | float] = self.prepare_result(batch_map, model_output)

        return result

    def predict_next(self, batch_map : BatchMap) -> torch.Tensor:
        self.net.eval()
        all_indices : torch.Tensor = torch.arange(0, self.net.embed.weight.shape[0])

        out : torch.Tensor = self.net.forward_probs(batch_map, all_indices)
        out = out[:, -1, :]

        return out


DioraName = Literal["treelstm", "mlp", "mlp-shared"]

class BuildNetOptions(TypedDict):
    lr : float
    hidden_dim : int
    inner_dim : int
    k_neg : int
    normalize : bool
    cuda : bool
    ngpus : int
    multigpu : bool
    local_rank : int
    ngpus : int
    master_addr : str
    master_port : str
    arch : DioraName
    load_model_path : Optional[str]
    experiment_name : str
    reconstruct_mode : Literal["margin", "softmax"]
    margin : int
    unk_idx : int
    word_embed_len : int
    char_embed_len : int


def build_net(options : BuildNetOptions) -> Trainer:

    logger = get_logger()

    lr : float = options["lr"]
    size : int = options["hidden_dim"]
    inner_size : int = options["inner_dim"]
    k_neg : int = options["k_neg"]
    normalize : bool = options["normalize"]
    cuda : bool = options["cuda"]
    multigpu : bool = options["multigpu"]
    local_rank : int | None = options["local_rank"]
    ngpus : int = options["ngpus"]
    word_embed_len : int = options["word_embed_len"]
    char_embed_len : int = options["char_embed_len"]


    # Embed
    embedding_layer : WordEmbedding | CharEmbedding = SimpleEmbedding(word_embed_len, size)
    #DualTransformerEmbed(len(word2i), len(char2i), size//2, size//2)
    #DualFixedEmbed(word2i, size//2, size//2) # TODO: Allow import of embeddings

    # Diora
    name_to_diora : Dict[DioraName, Type[Diora]] = {'treelstm' : DioraTreeLSTM,
                                                    'mlp' : DioraMLP,
                                                    'mlp-shared' : DioraMLPShared}

    diora : Diora = name_to_diora[options["arch"]](size,
                                                   inner_size = inner_size,
                                                   outside=True,
                                                   normalize=normalize,
                                                   compress=False)

    # Loss
    loss_funcs : List[LossFunc] = get_loss_funcs(options, embedding_layer, size)

    # Net
    net : Net
    net = Net(embedding_layer, diora, loss_funcs=loss_funcs, unk_index = options["unk_idx"])

    # Load model.
    if options["load_model_path"] is not None:
        logger.info('Loading model: {}'.format(options["load_model_path"]))
        Trainer.load_model(net, options["load_model_path"])

    # CUDA-support
    if cuda and not multigpu:
        diora.to("cuda")
        net.to("cuda")

    if cuda and multigpu:
        #torch.cuda.set_device(local_rank)

        print("Shifting model device to {}".format(local_rank))
        diora.to(local_rank)
        net.to(local_rank)
        embedding_layer.to(local_rank)
        print("Diora device:", diora.device)

        print("Initialising parallel on device {}".format(local_rank))
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [local_rank], output_device = local_rank)
        print("Initialised parallel on device {}".format(local_rank))

    # Trainer
    trainer = Trainer(net, k_neg=k_neg, ngpus=ngpus, cuda=cuda,
                      rank = local_rank, experiment_name = options["experiment_name"])
    trainer.init_optimizer(optim.Adam, dict(lr=lr, betas=(0.9, 0.999), eps=1e-8))

    return trainer
