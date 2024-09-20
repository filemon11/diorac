import torch
import torch.nn as nn

from diora.net.outside_index import get_outside_index
from diora.net.inside_index import get_inside_index
from diora.net.offset_cache import get_offset_cache

from typing import Literal, Optional, Dict, Tuple, List, Callable

TINY = 1e-8


class UnitNorm(object):
    def __call__(self, x : torch.Tensor, p : int = 2, eps : float = TINY) -> torch.Tensor:
        return x / x.norm(p=p, dim=-1, keepdim=True).clamp(min=eps)


class NormalizeFunc(nn.Module):
    def __init__(self, mode : Literal['none', 'unit'] = 'none'):
        super(NormalizeFunc, self).__init__()
        self.mode : Literal['none', 'unit'] = mode

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        mode = self.mode
        if mode == 'none':
            return x
        elif mode == 'unit':
            return UnitNorm()(x)


class BatchInfo(object):
    length : int
    level : int
    batch_size : int
    size : int
    def __init__(self, **kwargs):
        super(BatchInfo, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


class Chart(object):
    def __init__(self, batch_size : int, length : int, 
                 size : int, dtype : Optional[torch.dtype] = None, cuda : bool = False, device : Optional[str] = None):
        super(Chart, self).__init__()

        ncells : int = int(length * (1 + length) / 2)

        device = False if cuda is False else (torch.cuda.current_device() if device is None else device)

        ## Inside.
        self.inside_h : torch.Tensor = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
        self.inside_c : torch.Tensor = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
        self.inside_s : torch.Tensor = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)

        ## Outside.
        self.outside_h : torch.Tensor = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
        self.outside_c : torch.Tensor = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
        self.outside_s : torch.Tensor = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)


class Index(object):
    def __init__(self, cuda : bool = False):
        super(Index, self).__init__()
        self.cuda : bool = cuda
        self.inside_index_cache : Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.outside_index_cache : Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.offset_cache : Dict[int, int] = {}

    def get_offset(self, length : int) -> int:
        if length not in self.offset_cache:
            self.offset_cache[length] = get_offset_cache(length)
        return self.offset_cache[length]

    def get_inside_index(self, length : int, level, device) -> Tuple[int, int]:
        if (length, level) not in self.inside_index_cache:
            self.inside_index_cache[(length, level)] = \
                get_inside_index(length, level,
                    self.get_offset(length), cuda=self.cuda, device = device)
            
        return self.inside_index_cache[(length, level)]

    def get_outside_index(self, length : int, level : int, device = None) -> Tuple[int, int]:
        if (length, level) not in self.outside_index_cache:
            self.outside_index_cache[(length, level)] = \
                get_outside_index(length, level,
                    self.get_offset(length), cuda=self.cuda, device = device)
            
        return self.outside_index_cache[(length, level)]


# Composition Functions

class TreeLSTM(nn.Module):
    def __init__(self, size : int, inner_size : int | None = None, ninput : int = 2, leaf : bool = False):
        super(TreeLSTM, self).__init__()

        self.size : int = size
        self.inner_size : int = self.size if inner_size is None else inner_size
        self.ninput : int = ninput

        if leaf:
            self.W : nn.Parameter = nn.Parameter(torch.FloatTensor(3 * self.inner_size, self.size))
        self.U : nn.Parameter = nn.Parameter(torch.FloatTensor(5 * self.inner_size, self.ninput * self.size))
        self.B : nn.Parameter = nn.Parameter(torch.FloatTensor(5 * self.inner_size))
        self.O : nn.Parameter | None = nn.Parameter(torch.FloatTensor(self.size, self.inner_size)) if inner_size is not None else None
        self.I : nn.Parameter | None = nn.Parameter(torch.FloatTensor(self.inner_size, self.size)) if inner_size is not None else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in [p for p in self.parameters() if p.requires_grad]:
            param.data.normal_()

    def leaf_transform(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        W : nn.Parameter
        B : nn.Parameter
        W, B = self.W, self.B[:3*self.inner_size]

        #print("x shape", x.shape)

        activations : torch.Tensor = torch.matmul(x, W.t()) + B
        a_lst : List[torch.tensor] = torch.chunk(activations, 3, dim=-1)
        u : torch.Tensor = torch.tanh(a_lst[0])
        i : torch.Tensor = torch.sigmoid(a_lst[1])
        o : torch.Tensor = torch.sigmoid(a_lst[2])

        c : torch.Tensor = i * u
        h : torch.Tensor = o * torch.tanh(c)


        #print("h shape", h.shape)

        if self.O is not None:
            c = torch.matmul(c, self.O.t())
            h = torch.matmul(h, self.O.t())

        #print("h out shape", h.shape)
        return h, c

    def forward(self, hs : List[torch.Tensor],
                cs : List[torch.Tensor],
                constant : float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        U : nn.Parameter
        B : nn.Parameter
        U, B = self.U, self.B

        input_h : torch.Tensor = torch.cat(hs, 1)

        activations : torch.Tensor = torch.matmul(input_h, U.t()) + B
        a_lst : List[torch.Tensor] = torch.chunk(activations, 5, dim=1)
        u : torch.Tensor = torch.tanh(a_lst[0])
        i : torch.Tensor = torch.sigmoid(a_lst[1])
        o : torch.Tensor = torch.sigmoid(a_lst[2])
        f0 : torch.Tensor = torch.sigmoid(a_lst[3] + constant)
        f1 : torch.Tensor = torch.sigmoid(a_lst[4] + constant)
        
        cs0 : torch.Tensor = cs[0]
        cs1 : torch.Tensor = cs[1]
        if self.I is not None:
            cs0 = torch.matmul(cs0, self.I.t())
            cs1 = torch.matmul(cs1, self.I.t())

        #print("f0 shape", f0.shape, "cs0 shape", cs[0].shape)
        c : torch.Tensor = f0 * cs0 + f1 * cs1 + i * u
        h : torch.Tensor = o * torch.tanh(c)

        if self.O is not None:
            c = torch.matmul(c, self.O.t())
            h = torch.matmul(h, self.O.t())

        return h, c


class ComposeMLP(nn.Module):
    def __init__(self, size, inner_size : int | None = None, ninput=2, leaf=False):
        super(ComposeMLP, self).__init__()
        
        self.size = size
        self.inner_size = size if inner_size is None else inner_size
        self.ninput = ninput

        if leaf:
            self.V = nn.Parameter(torch.FloatTensor(self.size, self.size))
        self.W_0 = nn.Parameter(torch.FloatTensor(2 * self.size, self.inner_size))
        self.W_1 = nn.Parameter(torch.FloatTensor(self.inner_size, self.size))
        self.B = nn.Parameter(torch.FloatTensor(self.inner_size))
        self.B_1 = nn.Parameter(torch.FloatTensor(self.size))
        self.reset_parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        device = self.device
        return device.index is not None and device.index >= 0

    def reset_parameters(self):
        # TODO: Init with diagonal.
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def leaf_transform(self, x):
        h = torch.tanh(torch.matmul(x, self.V) + self.B_1)
        #print("Current device:", torch.cuda.current_device(), "Self.is_cuda:", self.is_cuda, "Self.device", self.device)
        device = self.device
        #device = torch.cuda.current_device() if self.is_cuda else None  #TODO
        c = torch.full(h.shape, 0, dtype=torch.float32, device=device)

        return h, c

    def forward(self, hs, cs, constant=1.0):
        #print("Component device:", self.device, "matrix device:", self.W_0.device, "Input device:", torch.cat(hs, 1).device)
        input_h = torch.cat(hs, 1)
        h = torch.relu(torch.matmul(input_h, self.W_0) + self.B)
        h = torch.relu(torch.matmul(h, self.W_1) + self.B_1)

        device = self.device
        #device = torch.cuda.current_device() if self.is_cuda else None
        c = torch.full(h.shape, 0, dtype=torch.float32, device=device)

        return h, c


class ComposeHeadMLP(nn.Module):    # TODO
    def __init__(self, size : int, ninput : int = 2, leaf : bool = False):
        super(ComposeHeadMLP, self).__init__()

        self.size : int = size
        self.ninput : int = ninput

        if leaf:
            self.V : nn.Parameter = nn.Parameter(torch.FloatTensor(self.size, self.size))

        self.W_H : nn.Parameter = nn.Parameter(torch.FloatTensor(self.size, self.size))
        self.W_0 : nn.Parameter = nn.Parameter(torch.FloatTensor(self.size, self.size))
        self.W_1 : nn.Parameter = nn.Parameter(torch.FloatTensor(self.size, self.size))

        #print(self.size)
        
        self.B : nn.Parameter = nn.Parameter(torch.FloatTensor(self.size))
        self.B_1 : nn.Parameter = nn.Parameter(torch.FloatTensor(self.size))

        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def is_cuda(self) -> bool:
        device : torch.device = self.device
        return device.index is not None and device.index >= 0

    def reset_parameters(self) -> None:
        # TODO: Init with diagonal.
        for param in [p for p in self.parameters() if p.requires_grad]:
            param.data.normal_()

    def leaf_transform(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h : torch.Tensor = torch.tanh(torch.matmul(x, self.V) + self.B)
        device = self.device
        #device : Optional[torch.device] = torch.cuda.current_device() if self.is_cuda else None
        c : torch.Tensor = torch.full(h.shape, 0, dtype=torch.float32, device=device)

        return h, c

    def forward(self, hs : List[torch.Tensor], 
                cs : List[torch.Tensor], 
                constant : float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        left : torch.Tensor = hs[0]
        right : torch.Tensor = hs[1]

        bma : torch.Tensor = torch.matmul(left, self.W_H).unsqueeze(1)
        ba : torch.Tensor = torch.matmul(bma, right.unsqueeze(2)).view(-1, 1)

        input_h : torch.Tensor = ba * left + (1 - ba) * right
        h : torch.Tensor = torch.relu(torch.matmul(input_h, self.W_0) + self.B)
        h : torch.Tensor = torch.relu(torch.matmul(h, self.W_1) + self.B_1)

        device = self.device
        #device : Optional[torch.device] = torch.cuda.current_device() if self.is_cuda else None
        c : torch.Tensor = torch.full(h.shape, 0, dtype=torch.float32, device=device)

        return h, c




# Score Functions

class Bilinear(nn.Module):
    def __init__(self, size : int):
        super(Bilinear, self).__init__()
        self.size : int = size
        self.mat : nn.Parameter = nn.Parameter(torch.FloatTensor(self.size, self.size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in [p for p in self.parameters() if p.requires_grad]:
            param.data.normal_()

    def forward(self, vector1 : torch.Tensor, vector2 : torch.Tensor) -> torch.Tensor:
        # bilinear
        # a = 1 (in a more general bilinear function, a is any positive integer)
        # vector1.shape = (b, m)
        # matrix.shape = (m, n)
        # vector2.shape = (b, n)
        bma : torch.Tensor = torch.matmul(vector1, self.mat).unsqueeze(1)
        ba : torch.Tensor = torch.matmul(bma, vector2.unsqueeze(2)).view(-1, 1)
        return ba


# Inside

def inside_fill_chart(batch_info : BatchInfo, chart : Chart, 
                      index : Index, h : torch.Tensor, c : torch.Tensor, s : torch.Tensor) -> None:
    L : int = batch_info.length - batch_info.level

    offset = index.get_offset(batch_info.length)[batch_info.level]

    chart.inside_h[:, offset:offset+L] = h
    chart.inside_c[:, offset:offset+L] = c
    chart.inside_s[:, offset:offset+L] = s


def get_inside_states(batch_info : BatchInfo, chart : torch.Tensor, index : Index, size : int, device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    lidx : int
    ridx : int
    lidx, ridx = index.get_inside_index(batch_info.length, batch_info.level, device = device)

    ls : torch.Tensor = chart.index_select(index=lidx, dim=1).view(-1, size)   # TODO
    rs : torch.Tensor = chart.index_select(index=ridx, dim=1).view(-1, size)

    return ls, rs


def inside_compose(compose_func : Callable[[List[torch.Tensor], List[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]],
                   hs : List[torch.Tensor], cs : List[torch.Tensor]) -> torch.Tensor:
    return compose_func(hs, cs)


def inside_score(score_func : Callable[[List[torch.Tensor], List[torch.Tensor]], torch.Tensor], 
                 batch_info : BatchInfo, hs : List[torch.Tensor], ss : List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    B : int = batch_info.batch_size
    L : int = batch_info.length - batch_info.level
    N : int = batch_info.level

    s : torch.Tensor = score_func(hs[0], hs[1]) + ss[0] + ss[1]
    s = s.view(B, L, N, 1)
    p : torch.Tensor = torch.softmax(s, dim=2)

    return s, p


def inside_aggregate(batch_info : BatchInfo, h : torch.Tensor, 
                     c : torch.Tensor, s : torch.Tensor, p : torch.Tensor, 
                     normalize_func : Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B : int = batch_info.batch_size
    L : int = batch_info.length - batch_info.level
    N : int = batch_info.level

    h_agg : torch.Tensor = torch.sum(h.view(B, L, N, -1) * p, 2)
    c_agg : torch.Tensor = torch.sum(c.view(B, L, N, -1) * p, 2)
    s_agg : torch.Tensor = torch.sum(s * p, 2)

    h_agg : torch.Tensor = normalize_func(h_agg)
    c_agg : torch.Tensor = normalize_func(c_agg)

    return h_agg, c_agg, s_agg


def inside_func(compose_func : Callable[[List[torch.Tensor], List[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]],
                score_func : Callable[[torch.Tensor], torch.Tensor], batch_info : BatchInfo, 
                chart : Chart, index : Index, 
                normalize_func : Callable[[torch.Tensor], torch.Tensor],
                device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO

    hlst : List[torch.Tensor] = list(get_inside_states(batch_info, chart.inside_h, index, batch_info.size, device = device))
    clst : List[torch.Tensor] = list(get_inside_states(batch_info, chart.inside_c, index, batch_info.size, device = device))
    slst : List[torch.Tensor] = list(get_inside_states(batch_info, chart.inside_s, index, 1, device = device))

    h : torch.Tensor
    c : torch.Tensor
    h, c = inside_compose(compose_func, hlst, clst)
    
    s : torch.Tensor
    p : torch.Tensor
    s, p = inside_score(score_func, batch_info, hlst, slst)

    hbar : torch.Tensor
    cbar : torch.Tensor
    sbar : torch.Tensor
    hbar, cbar, sbar = inside_aggregate(batch_info, h, c, s, p, normalize_func)

    inside_fill_chart(batch_info, chart, index, hbar, cbar, sbar)

    return h, c, s


# Outside

def outside_fill_chart(batch_info : BatchInfo, chart : Chart, 
                       index : Index, h : torch.Tensor, c : torch.Tensor, s : torch.Tensor) -> None:
    L : int = batch_info.length - batch_info.level

    offset : int = index.get_offset(batch_info.length)[batch_info.level]

    chart.outside_h[:, offset:offset+L] = h
    chart.outside_c[:, offset:offset+L] = c
    chart.outside_s[:, offset:offset+L] = s


def get_outside_states(batch_info : BatchInfo, pchart, schart, index : Index, size : int, device = None):  #TODO
    pidx : int
    sidx : int
    pidx, sidx = index.get_outside_index(batch_info.length, batch_info.level, device = device)

    ps = pchart.index_select(index=pidx, dim=1).view(-1, size)
    ss = schart.index_select(index=sidx, dim=1).view(-1, size)

    return ps, ss


def outside_compose(compose_func : Callable[[List[torch.Tensor], List[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]], 
                    hs : List[torch.Tensor], 
                    cs : List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return compose_func(hs, cs, 0)  #TODO


def outside_score(score_func, batch_info, hs, ss):
    B = batch_info.batch_size
    L = batch_info.length - batch_info.level

    s = score_func(hs[0], hs[1]) + ss[0] + ss[1]
    s = s.view(B, -1, L, 1)
    p = torch.softmax(s, dim=1)

    return s, p


def outside_aggregate(batch_info, h, c, s, p, normalize_func):
    B = batch_info.batch_size
    L = batch_info.length - batch_info.level
    N = s.shape[1]

    h_agg = torch.sum(h.view(B, N, L, -1) * p, 1)
    c_agg = torch.sum(c.view(B, N, L, -1) * p, 1)
    s_agg = torch.sum(s * p, 1)

    h_agg = normalize_func(h_agg)
    c_agg = normalize_func(c_agg)

    return h_agg, c_agg, s_agg


def outside_func(compose_func, score_func, batch_info, chart, index, normalize_func, device = None):
    ph, sh = get_outside_states(
        batch_info, chart.outside_h, chart.inside_h, index, batch_info.size, device = device)
    pc, sc = get_outside_states(
        batch_info, chart.outside_c, chart.inside_c, index, batch_info.size, device = device)
    ps, ss = get_outside_states(
        batch_info, chart.outside_s, chart.inside_s, index, 1, device = device)

    hlst = [sh, ph]
    clst = [sc, pc]
    slst = [ss, ps]

    h, c = outside_compose(compose_func, hlst, clst)
    s, p = outside_score(score_func, batch_info, hlst, slst)
    hbar, cbar, sbar = outside_aggregate(batch_info, h, c, s, p, normalize_func)

    outside_fill_chart(batch_info, chart, index, hbar, cbar, sbar)

    return h, c, s


# Base

class DioraBase(nn.Module):
    r"""DioraBase

    """

    def __init__(self, size, inner_size : int | None = None, outside=True, normalize='unit', compress=False):
        super(DioraBase, self).__init__()
        assert normalize in ('none', 'unit'), 'Does not support "{}".'.format(normalize)

        self.size = size
        self.inner_size : int | None = inner_size
        self.outside = outside
        self.inside_normalize_func = NormalizeFunc(normalize)
        self.outside_normalize_func = NormalizeFunc(normalize)
        self.compress = compress
        self.ninput = 2

        self.index = None

        self.init_parameters()
        self.reset_parameters()
        self.reset()

    def init_parameters(self):
        raise NotImplementedError

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def inside_h(self):
        return self.chart.inside_h

    @property
    def inside_c(self):
        return self.chart.inside_c

    @property
    def inside_s(self):
        return self.chart.inside_s

    @property
    def outside_h(self):
        return self.chart.outside_h

    @property
    def outside_c(self):
        return self.chart.outside_c

    @property
    def outside_s(self):
        return self.chart.outside_s

    @property
    def is_cuda(self):
        device = self.device
        return device.index is not None and device.index >= 0

    def cuda(self):
        super(DioraBase, self).cuda()
        if self.index is not None:
            self.index.cuda = True # TODO: Should support to/from cpu/gpu.
    
    def to(self, device):
        super(DioraBase, self).to(device)
        if self.index is not None:
            self.index.cuda = True

    def get(self, chart, level):
        length = self.length
        L = length - level
        offset = self.index.get_offset(length)[level]
        return chart[:, offset:offset+L]

    def leaf_transform(self, x):
        normalize_func = self.inside_normalize_func
        transform_func = self.inside_compose_func.leaf_transform

        input_shape = x.shape[:-1]
        #print("input_shape", input_shape)
        h, c = transform_func(x)
        #print("h, c devices:", h.device, c.device)
        h = normalize_func(h.view(*input_shape, self.size))
        c = normalize_func(c.view(*input_shape, self.size))

        return h, c

    def inside_pass(self):
        compose_func = self.inside_compose_func
        score_func = self.inside_score_func
        index = self.index
        chart = self.chart
        normalize_func = self.inside_normalize_func

        for level in range(1, self.length):

            batch_info = BatchInfo(
                batch_size=self.batch_size,
                length=self.length,
                size=self.size,
                level=level,
                )

            h, c, s = inside_func(compose_func, score_func, batch_info, chart, index,
                normalize_func=normalize_func, device = self.device)

            self.inside_hook(level, h, c, s)

    def inside_hook(self, level, h, c, s):
        pass

    def outside_hook(self, level, h, c, s):
        pass

    def initialize_outside_root(self):
        B = self.batch_size
        D = self.size
        normalize_func = self.outside_normalize_func

        if self.compress:
            h = torch.matmul(self.inside_h[:, -1:], self.root_mat_out)
        else:
            h = self.root_vector_out_h.view(1, 1, D).expand(B, 1, D)
        if self.root_vector_out_c is None:
            device = self.device
            #device = torch.cuda.current_device() if self.is_cuda else None
            c = torch.full(h.shape, 0, dtype=torch.float32, device=device)
        else:
            c = self.root_vector_out_c.view(1, 1, D).expand(B, 1, D)

        h = normalize_func(h)
        c = normalize_func(c)

        self.chart.outside_h[:, -1:] = h
        self.chart.outside_c[:, -1:] = c

    def outside_pass(self):
        self.initialize_outside_root()

        compose_func = self.outside_compose_func
        score_func = self.outside_score_func
        index = self.index
        chart = self.chart
        normalize_func = self.outside_normalize_func

        for level in range(self.length - 2, -1, -1):
            batch_info = BatchInfo(
                batch_size=self.batch_size,
                length=self.length,
                size=self.size,
                level=level,
                )

            h, c, s = outside_func(compose_func, score_func, batch_info, chart, index,
                normalize_func=normalize_func, device = self.device)

            self.outside_hook(level, h, c, s)

    def init_with_batch(self, h, c):
        size = self.size
        batch_size, length, _ = h.shape

        self.batch_size = batch_size
        self.length = length

        self.chart = Chart(batch_size, length, size, dtype=torch.float32, cuda=self.is_cuda, device = self.device)
        self.chart.inside_h[:, :self.length] = h
        self.chart.inside_c[:, :self.length] = c

    def reset(self):
        self.batch_size = None
        self.length = None
        self.chart = None

    def get_chart_wrapper(self):
        return self

    def forward(self, x):
        if self.index is None:
            self.index = Index(cuda=self.is_cuda)

        #print("In-DIORA device:", x.device)
        
        self.reset()
        #print("x shape:", x.shape)
        h, c = self.leaf_transform(x)
        #print("h shape:", h.shape)
        #print("DIORA leaves:", h.device, c.device)  # for the second one, h is 1 and c is 0 (but should be 1)
        self.init_with_batch(h, c)

        self.inside_pass()

        if self.outside:
            self.outside_pass()

        return None


class DioraTreeLSTM(DioraBase):
    r"""DioraTreeLSTM

    """

    def init_parameters(self):
        # Model parameters for transformation required at both input and output
        self.inside_score_func = Bilinear(self.size)
        self.outside_score_func = Bilinear(self.size)

        if self.compress:
            self.root_mat_out = nn.Parameter(torch.FloatTensor(self.size, self.size))
        else:
            self.root_vector_out_h = nn.Parameter(torch.FloatTensor(self.size))
        self.root_vector_out_c = nn.Parameter(torch.FloatTensor(self.size))

        self.inside_compose_func = TreeLSTM(self.size, self.inner_size, ninput=self.ninput, leaf=True)
        self.outside_compose_func = TreeLSTM(self.size, self.inner_size, ninput=2, leaf=False)


class DioraMLP(DioraBase):

    def init_parameters(self):
        self.inside_score_func = Bilinear(self.size)
        self.outside_score_func = Bilinear(self.size)

        if self.compress:
            self.root_mat_out = nn.Parameter(torch.FloatTensor(self.size, self.size))
        else:
            self.root_vector_out_h = nn.Parameter(torch.FloatTensor(self.size))
        self.root_vector_out_c = None

        self.inside_compose_func = ComposeMLP(self.size, self.inner_size, leaf=True)
        self.outside_compose_func = ComposeMLP(self.size, self.inner_size)


class DioraMLPShared(DioraBase):

    def init_parameters(self):
        self.inside_score_func = Bilinear(self.size)
        self.outside_score_func = self.inside_score_func

        if self.compress:
            self.root_mat_out = nn.Parameter(torch.FloatTensor(self.size, self.size))
        else:
            self.root_vector_out_h = nn.Parameter(torch.FloatTensor(self.size))
        self.root_vector_out_c = None

        self.inside_compose_func = ComposeMLP(self.size, self.inner_size, leaf=True)
        self.outside_compose_func = self.inside_compose_func
