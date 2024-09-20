"""A collection of utilities to split `diora.data.representation.Representation`
objects into subunits. These are not called tokenizers since they
do not support assigning indices to tokens, in contrast to
the `transformers.AutoTokenizer`s.
Tree tokenization logic is adapted from 
https://github.com/OmarMomen14/Linguistic-Structure-Induction-from-Language-Models/

Classes
----------
Segmenter
    Basic segmenter for string representations
    that splits the string into subunits.
TreeSegmenter
    Extension to `Segmenter` that allows to
    tokenize tree representations.
Word2i
    Mapping from tokens to indices.
FutureWord2i
    Mapping from tokens to indices that supports
    special tokens.
Char2i
    Mapping from characters to indices. Supports
    automatically splitting strings into individual
    characters.
"""

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from diora.data.representations import Tree, TreeRepresentation, Representation
from diora.data.representations import TokenizedRepresentation, FlatTokenizedRepresentation

from collections import defaultdict

from typing import Union, Optional, List, Tuple, Dict, DefaultDict, Iterator
from typing import overload, cast, Generic, TypeVar, Literal, Callable
from typing_extensions import Self, override

from abc import ABC, abstractmethod


MAX_LEN = 512
"""Maximum length for tokenization."""

FUTURE_TOKEN = "<f>"
"""Token that represents the possible continuation
of a sentence."""

BOS_TOKEN = "<s>"
"""Functional sentence beginning token."""

EOS_TOKEN = "</s>"
"""Functional sentence end token."""

UNK_TOKEN = "<unk>"
"""Token that represents unknown tokens."""

SEP_TOKEN = "<sep>"
"""Special token that represents sentence separations."""

PAD_TOKEN = "<pad>"
"""Special token that can be used for padding sentences."""

CLS_TOKEN = "<cls>"
"""Special token representing the class of the input."""

MASK_TOKEN = "<mask>"
"""Special token representing a masked token."""

RP = TypeVar("RP", bound = str)
TRP = TypeVar("TRP", bound = TokenizedRepresentation)
TR = TypeVar("TR", bound = TreeRepresentation)
T = TypeVar("T", bound = Tree)
W = TypeVar("W", bound = "Word2i")
C = TypeVar("C", bound = "Char2i")

TokenizerMode = Literal["simple", "word", "sentence", "fragment"] 
"""Tokenization mode. Decides whether to adjoin special labels."""

class _Segmenter(Generic[RP, TRP], ABC):
    """This abstract class defines a segmenter
    as a callable object defining a mapping from
    representations to tokenized representations."""
    @abstractmethod
    def __call__(self, sequence : Representation,
                 mode : TokenizerMode = "simple",
                 add_future : bool = False,
                 start_position : int = 0) -> TRP:
        """Should produce a tokenized representation
        from a representation.

        Parameters
        ----------
        sentence : RP
            A representation.
        as_sentence : bool, optional
            Whether to prepend a sentence start token
            and append a sentence end token.
        add_future : bool, optional
            Whether to append a token representing
            a possible continuation of the sentence.

        Returns
        -------
        TRP
            The resulting tokenized representation.
        """
        ...


class Segmenter(_Segmenter[str, FlatTokenizedRepresentation]):
    """
    Tokenizes character sequences.
    Wraps around huggingface tokenizer.

    Attributes
    -------
    tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast
        Huggingface tokenizer.
    """

    def __init__(self, tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast) -> None:
        """Initialise segmenter.

        Parameters
        ----------
        tokenizer : AutoTokenizer, defaults to None
            A huggingface tokenizer. If None, then simple
            setting (splitting at delimiter) is used.
        """

        self.tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer
        """Huggingface tokenizer."""

    @property
    def unk_token(self) -> str:
        return self.tokenizer.unk_token

    @property
    def bos_token(self) -> str:
        return self.tokenizer.bos_token

    @property
    def eos_token(self) -> str:
        return self.tokenizer.eos_token
    
    @property
    def sep_token(self) -> str:
        return self.tokenizer.sep_token
    
    @property
    def pad_token(self) -> str:
        return self.tokenizer.pad_token
    
    @property
    def cls_token(self) -> str:
        return self.tokenizer.cls_token
    
    @property
    def mask_token(self) -> str:
        return self.tokenizer.mask_token

    @property
    def future_token(self) -> str:
        return self.tokenizer.special_tokens_map["additional_special_tokens"][0]
    
    @property
    def word2i(self) -> Dict[str, int]:
        """Load word2i from huggingface
        tokenizer.

        Returns
        -------
        Dict[str, int]
            Word2i dictionary extracted from
            the huggingface tokenizer.
        """
        return self.tokenizer.vocab
        
    @classmethod
    def from_pretrained(cls, name : str,
                        bos_token : str = BOS_TOKEN,
                        eos_token : str = EOS_TOKEN,
                        unk_token : str = UNK_TOKEN,
                        sep_token : str = SEP_TOKEN,
                        pad_token : str = PAD_TOKEN,
                        cls_token : str = CLS_TOKEN,
                        mask_token : str = MASK_TOKEN,
                        future_token : str = FUTURE_TOKEN) -> Self:
        """Load a pretrained huggingface `AutoTokenizer`.

        Parameters
        ----------
        name : str
            Name of the tokenizer.
        bos_token : str
        eos_token : str
        unk_token : str
        sep_token : str
        pad_token : str
        cls_token : str
        mask_token : str
        future_token : str

        Returns
        -------
        Self
            Segmenter object.
        """
        tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast 
        tokenizer = AutoTokenizer.from_pretrained(name)

        tokenizer.add_special_tokens({"bos_token" : bos_token,
                                      "eos_token" : eos_token,
                                      "unk_token" : unk_token,
                                      "sep_token" : sep_token,
                                      "pad_token" : pad_token,
                                      "cls_token" : cls_token,
                                      "mask_token" : mask_token,
                                      "additional_special_tokens" : [future_token]},
                                      replace_additional_special_tokens = True)

        return cls(tokenizer)
    
    def tokenize(self, sequence : str, 
                 mode : TokenizerMode = "simple",
                 add_future : bool = False,
                 start_position : int = 0) -> FlatTokenizedRepresentation:
        """Split a character sequence into subsequences
        using a tokenizer. Keeps track of which resuilting
        token corresponds to which source word via index markers.
        
        Options for `mode` argument:
            - 'simple': does not adjoin any special characters.
            - 'word': does not adjoin any special characters.
              Returns the `start_position` parameter n times
              repeated for a token sequence of n.
            - 'sentence': prepends beginning of sentence token.
              Appends end of sentence token.
            - 'fragment': Only prepends beginning of sentence
              token.

        Parameters
        ----------
        sequence : str
            The sequence to split.
        mode : Literal['simple', 'word', 
                       'sentence', 'fragment'], defaults to 'simple'
            Special character and index marker mode.
        add_future : bool, defaults to False
            Whether to append the special future token.
        start_position : int
            The index marker that should be assigned to the
            first resulting token.

        Returns
        -------
        FlatTokenizedRepresentation
            The resulting representation
        """


        tokens : List[str]
        markers : List[int]

        result : BatchEncoding = self.tokenizer(sequence, 
                                                max_length = MAX_LEN, 
                                                truncation = True)
        
        ids : List[int] = result["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(ids)

        markers = [-1 if i == None else i for i in result.word_ids()]
        markers = [(i + start_position if i >= 0 else i) for i in markers]

        match mode:
            case "simple":
                tokens = tokens[1:-1]
                markers = markers[1:-1]
            
            case "word":
                tokens = tokens[1:-1]
                markers = [start_position] * len(tokens)

            case "sentence":
                pass

            case "fragment":
                tokens = tokens[:-1]
                markers = markers[:-1]

        if add_future:
            tokens.append(self.future_token)
            markers.append(-1)

        return FlatTokenizedRepresentation(None, tokens, markers)
    
    def __call__(self, sequence : Representation,
                 mode : Literal["simple", "word", "sentence", "fragment"] = "simple",
                 add_future : bool = False,
                 start_position : int = 0) -> FlatTokenizedRepresentation:
        """Split a character sequence into subsequences
        using a tokenizer. Keeps track of which resuilting
        token corresponds to which source word via index markers.
        
        Options for `mode` argument:
            - 'simple': does not adjoin any special characters.
            - 'word': does not adjoin any special characters.
              Returns the `start_position` parameter n times
              repeated for a token sequence of n.
            - 'sentence': prepends beginning of sentence token.
              Appends end of sentence token.
            - 'fragment': Only prepends beginning of sentence
              token.

        Parameters
        ----------
        sequence : str
            The sequence to split.
        mode : Literal['simple', 'word', 
                       'sentence', 'fragment'], defaults to 'simple'
            Special character and index marker mode.
        add_future : bool, defaults to False
            Whether to append the special future token.
        start_position : int
            The index marker that should be assigned to the
            first resulting token.

        Returns
        -------
        FlatTokenizedRepresentation
            The resulting representation
        """
        
        return self.tokenize(sequence, mode, add_future, start_position)
    

class TreeSegmenter(Segmenter, _Segmenter[TR, TR]):
    """
    Extension to the `Segmenter` class that
    allows to tokenize trees representations.

    Since trees already express a tokenization,
    further splits driven by this class are treated
    as subword-tokenizations. Affected leaves are
    transformed into flat subtrees accordingly.
    """
    
    def tokenize_tree(self, tree : TR,
                      mode : Literal["simple", "word", "sentence", "fragment"] = "simple",
                      add_future : bool = False,
                      start_position : int = 0,
                      inplace : bool = False) -> TR:
        """Split a tree representation into subsequences
        using a tokenizer. Treats resulting subsequences
        as subwords. Affected leaves are transformed into flat 
        subtrees accordingly. Keeps track of which token corresponds 
        to which source word by maintaining the start index
        attributes of the tree nodes.

        Options for `mode` argument:
            - 'simple': does not adjoin any special characters.
            - 'word': does not adjoin any special characters.
              Returns the `start_position` parameter n times
              repeated for a token sequence of n.
            - 'sentence': prepends beginning of sentence token.
              Appends end of sentence token.
            - 'fragment': Only prepends beginning of sentence
              token.

        Special tokens (sentence beginning, sentence end,
        future token) are attached to the root node of
        the tree.

        Parameters
        ----------
        tree : str
            The tree to tokenize.
        mode : Literal['simple', 'word', 
                       'sentence', 'fragment'], defaults to 'simple'
            Special character and index marker mode.
        add_future : bool, defaults to False
            Whether to append the special future token.
        start_position : int
            The index marker that should be assigned to the
            first resulting token.
        inplace : bool, defaults to False
            Whether to perform the tree transformation
            inplace. Still creates a new representation object.

        Returns
        -------
        TR
            Resulting tree representation. Is a new object.
        """

        def _prepend_token(tree : Tree, token : str, label : str, index : int) -> None:
            """Prepend a token to a tree. A new leaf is created that is
            attached to the tree root."""
            tree.insert(0, tree.__class__(label, [token], index))
        
        def _append_token(tree : Tree, token : str, label : str, index : int) -> None:
            """Append a token to a tree. A new leaf is created that is
            attached to the tree root."""
            tree.append(tree.__class__(token, [label], index))

        # retrieve tree
        out_tree : Tree
        if inplace:
            self._tokenize_tree(tree.tree, 
                                flag = True, 
                                inplace = True)
            out_tree = tree.tree
        
        else:
            out_tree = self._tokenize_tree(tree.tree, 
                                           flag = True, 
                                           inplace = False)[0]

        # checkout modes
        match mode:
            case "simple":
                pass

            case "word":
                pass

            case "sentence":
                _prepend_token(out_tree, self.bos_token, self.bos_token, -1)
                _append_token(out_tree, self.eos_token, self.eos_token, -1)

            case "fragment":
                _prepend_token(out_tree, self.bos_token, self.bos_token, -1)

        # add special future token
        if add_future:
            _append_token(out_tree, self.future_token, self.future_token, -1)

        # calibrate start position
        index_to_add : int = start_position - out_tree.start_idx    # i.e. the difference to add to
                                                                    # every node index to shift the
                                                                    # whole tree right

        idx_change_needed : List[Tree] = [out_tree]
        node : Tree
        while len(idx_change_needed) != 0:
            node = idx_change_needed.pop()

            if isinstance(node, str):
                continue

            idx_change_needed.extend(node)

            if node.start_idx != -1:        # special token indices are not altered
                if mode == "word":
                    # overwrite if the tree represents a single word
                    node.start_idx = start_position
                else:
                    node.start_idx += index_to_add

        return tree.__class__(None, out_tree)
    
    @overload
    def _tokenize_tree(self, tree : T, flag : bool,
                       inplace : Literal[True]) -> Tuple[None, bool]:
        ...
    
    @overload
    def _tokenize_tree(self, tree : T, flag : bool = True,
                       inplace : Literal[False] = False) -> Tuple[T, bool]:
        ...
        
    def _tokenize_tree(self, tree : T, flag : bool = True,
                       inplace : bool = False) -> Tuple[Optional[T], bool]:
        """Converts tree into tree with subword
        tokens as new leaves.
        
        Parameters
        ----------
        tree : Tree
            The tree to convert.
        flag : bool, defaults to True
            If False prepends a blank space
            to the leaf that represents
            a word break.
        inplace : bool, defaults to False
            Whether to perform the tree
            transformations inplace.

        Returns
        -------
        new_tree : Optional[TR]
            Resulting tokenized tree. If `inplace = True`,
            `new_tree` is None.
        flag
            Whether the tree's rightmost leaf
            was a token.    # TODO: really?
        """

        new_children : List[Tree]

        if len(tree) == 1 and isinstance(tree[0], str):
            tokens : List[str]
            
            if flag:
                tokens = self.tokenize(tree[0], mode = "word").tokens
            else:
                tokens = self.tokenize(' '+tree[0], mode = "word").tokens
            
            flag = False

            if len(tokens) == 1:
                if inplace:
                    tree[0] = tokens[0]
                else:
                    new_children = [tokens[0]]
            
            else:
                new_children = [tree.__class__("s" + tree.label(), 
                                               [token], 
                                               tree.start_idx) 
                                                        for token in tokens]
                
                if inplace:
                    tree[0] = new_children[0]
                    tree.extend(new_children[1:])

        else:
            if inplace:
                for child in tree:
                    _, flag = self._tokenize_tree(child, flag, inplace)
            
            else:
                new_children = []
                for child in tree:
                    new_child, flag = self._tokenize_tree(child, flag, inplace)
                    new_children.append(new_child)

        if inplace:
            return None, flag
        
        else:
            return tree.__class__(tree.label(), new_children, tree.start_idx), flag
        
    
    @overload
    def __call__(self, sequence : TR,
                 mode : Literal["simple", "word", "sentence", "fragment"] = "simple",
                 add_future : bool = False,
                 start_position : Optional[int] = None,
                 inplace : bool = False) \
                    -> TR:
        ...

    @overload
    def __call__(self, sequence : str, # type: ignore
                 mode : Literal["simple", "word", "sentence", "fragment"] = "simple",
                 add_future : bool = False,
                 start_position : Optional[int] = None,
                 inplace : bool = False) \
                    -> FlatTokenizedRepresentation:
        ...

    @override
    def __call__(self, sequence : Union[TR, str],
                 mode : Literal["simple", "word", "sentence", "fragment"] = "simple",
                 add_future : bool = False,
                 start_position : Optional[int] = None,
                 inplace : bool = False) \
                    -> Union[TR, FlatTokenizedRepresentation]:
        """Tokenize a flat representation or a TreeRepresentation.
        
        - For trees:
        Split a tree representation into subsequences
        using a tokenizer. Treats resulting subsequences
        as subwords. Affected leaves are transformed into flat 
        subtrees accordingly. Keeps track of which token corresponds 
        to which source word by maintaining the start index
        attributes of the tree nodes.
        Special tokens (sentence beginning, sentence end,
        future token) are attached to the root node of
        the tree.

        - For flat representations:
        Produces a `FlatTokenizedRepresentation`.

        Options for `mode` argument:
            - 'simple': does not adjoin any special characters.
            - 'word': does not adjoin any special characters.
              Returns the `start_position` parameter n times
              repeated for a token sequence of n.
            - 'sentence': prepends beginning of sentence token.
              Appends end of sentence token.
            - 'fragment': Only prepends beginning of sentence
              token.

        Parameters
        ----------
        tree : str
            The tree to tokenize.
        mode : Literal['simple', 'word', 
                       'sentence', 'fragment'], defaults to 'simple'
            Special character and index marker mode.
        add_future : bool, defaults to False
            Whether to append the special future token.
        start_position : Optional[int], defaults to None
            The index marker that should be assigned to the
            first resulting token. If None: use 0 for 
            flat representations and retain existing indices
            in tree representations.
        inplace : bool, defaults to False
            Whether to perform the tree transformation
            inplace. Still creates a new representation object.
            Throws an exception if True while inputting
            a non-tree representation.

        Returns
        -------
        TR | FlatTokenizedRepresentation
            Resulting tree representation or flat tokenized
            representation depending on the input.
        """

        if isinstance(sequence, TreeRepresentation):
            if start_position is None:
                start_position = sequence.tree.start_idx

            return self.tokenize_tree(sequence, mode, add_future, start_position, inplace)
        
        else:
            assert inplace is not True, "Cannot perform tokenization inplace for non-tree representation."

            if start_position is None:
                start_position = 0

            return super().tokenize(sequence, mode, add_future, start_position)
        

class Word2i(Dict[str, int]):
    """Converts tokens to indices.
    Acts as defaultdict, assigning a special
    unk_token to tokens that are not in the
    word2i dictionary.

    Attributes
    ----------
    tokenizer_vocab : Optional[Dict[str, int]]
        Vocabulary retrieved from tokenizer
        if given at initialisation and not used
        directly for the Word2i mapping.
    default_factory : Callable[[], int]
        Produces the default value for
        unknown tokens.
    token_count : DefaultDict[str, int]
        Records the number of times
        a token has been seen during
        initialisation.
    """
    def __init__(self, segmenter : Optional[Segmenter] = None,
                 init_add : bool = False,
                 _mapping : Optional[Dict[str, int]] = None,
                 _default : int = 0) -> None:
        """Initialises a mapping from tokens to indices.
        If a segmenter is provided, it takes its `unk_token`
        index as the default index for the mapping.

        Parameters
        ----------
        segmenter : Optional[Segmenter], defaults to None
            Segmenter to retrieve indices for tokens from.
            If None, indices are assigned incrementally.
        init_add : bool, defaults to False
            If True: takes the vocabulary from the tokenizer
            (if provided) at initialisation. If False:
            takes goal indices from tokenizer and adds them
            to the vocabulary manually when calling `.add[token]`.
        """
                 
        self.tokenizer_vocab : Optional[Dict[str, int]] = None
        """Pointer to the tokenizer vocabulary if not initially
        added to the dictionary."""

        self.default_factory : Callable[[], int]
        """Produces the defalt value for unknow tokens"""

        self.token_count : DefaultDict[str, int] = defaultdict(int)
        """Records the number of times a token has been seen during
        initialisation.
        """
        
        if segmenter is not None:
            if _mapping is not None:
                raise Exception("Cannot provide tokenizer and _mapping at the same time.")

            if init_add:
                # Take vocabulary from the tokenizer
                super().__init__(segmenter.tokenizer.vocab)
            
            else:
                self.tokenizer_vocab = segmenter.tokenizer.vocab
                super().__init__()
            
            unk_idx : int = segmenter.tokenizer.vocab[segmenter.unk_token]

            self.default_factory = lambda : unk_idx
            
        else:
            if _mapping is not None:
                # Manual initialisation
                super().__init__(_mapping)
        
            else:
                # Initialise as empty mapping
                super().__init__()
            
            self.default_factory = lambda : _default
    
    @override
    def __len__(self) -> int:
        """Retrieves the necessary dimensionality of
        a word embedding to accomodate for the
        mapping, i.e. the highest index in the mapping's
        values + 1.

        Returns
        -------
        int
            Highest index + 1.
        """
        assert self.default_factory is not None

        if len(self.values()) > 0:
            return max(self.default_factory(), max(self.values())) + 1
        
        else:
            # only default value exists
            return self.default_factory() + 1

    @overload
    def __getitem__(self, token : str) -> int:
        ...

    @overload
    def __getitem__(self, token : List[str]) -> List[int]:
        ...

    @overload
    def __getitem__(self, token : List[List[str]]) -> List[List[int]]:
        ...

    @override
    def __getitem__(self, token : Union[str, List[str], List[List[str]]]) -> Union[int, List[int], List[List[int]]]:
        """Converts token(s) to indices.
        Does not alter the underlying mapping.

        Parameters
        ----------
        token : Union[str, List[str], List[List[str]]]
            A single token, a list of tokens or
            a list of lists of tokens.

        Returns
        -------
        Union[int, List[int], List[List[int]]]
            The corresponding (list (of lists) of) index/indices.
        """
        return self.get_idx(token, add_to_dict = False)
            
    @overload
    def get_idx(self, token : str, add_to_dict : bool = False) -> int:
        ...

    @overload
    def get_idx(self, token : List[str], add_to_dict : bool = False) -> List[int]:
        ...

    @overload
    def get_idx(self, token : List[List[str]], add_to_dict : bool = False) -> List[List[int]]:
        ...

    def get_idx(self, token : Union[str, List[str], List[List[str]]],
                    add_to_dict : bool = False) -> Union[int, List[int], List[List[int]]]:
        """Converts token(s) to indices.
        Can be used to add unseen tokens
        to the mapping.

        Parameters
        ----------
        token : Union[str, List[str], List[List[str]]]
            A single token, a list of tokens or
            a list of lists of tokens.
        add_to_dict : bool, defaults to False
            Whether to add tokens to the
            mapping if they are not there already.

        Returns
        -------
        Union[int, List[int], List[List[int]]]
            The corresponding (list (of lists) of) index/indices.
        """
        
        if isinstance(token, str):
            return self._retrieve(token, add_to_dict)
        
        else:
            if isinstance(token, list) and (len(token) == 0 or isinstance(token[0], str)):
                token = cast(List[str], token)
                return [self._retrieve(t, add_to_dict) for t in token]
            
            else:
                token = cast(List[List[str]], token)
                out_list : List[List[int]] = []
                for i, word in enumerate(token):

                    out_list.append([self._retrieve(t, add_to_dict) for t in word])
                return out_list
            
    def _retrieve(self, token : str, add_to_dict : bool) -> int:
        """Converts a single token to its index.

        Parameters
        ----------
        token : str
            A single token.
        add_to_dict : bool, defaults to False
            Whether to add the token to the
            mapping if it is not in it already.

        Returns
        -------
        int
            The corresponding index.
        """

        if add_to_dict:
            # Increase token count if currently in initialisation.
            self.token_count[token] += 1

        try:
            return super().__getitem__(token)
        
        except KeyError:

            if add_to_dict:

                value : int

                if self.tokenizer_vocab is None:
                    # If tokenizer was not provided, simply
                    # adding the next higher number.
                    value = len(self)
                    super().__setitem__(token, value)
                    
                    return value

                else:
                    # If tokenizer was provided, take
                    # index from it.

                    try:
                        # Only add token if it is present in the
                        # tokenizer vocabulary
                        value = self.tokenizer_vocab[token]
                        super().__setitem__(token, value)

                        return value

                    except KeyError:
                        pass

            return self.default_factory()
        
    def del_rare(self, count : int) -> None:
        """Deletes tokens that have been seen
        only `count` times or less during 
        initialisation. Thus, they are assigned
        the unknown token from now on.

        Parameters
        ----------
        count : int
            Tokens seen this many times or less
            are replaced with unknown from now on.
        """
        
        deleted_num : int = 0

        for token, token_count in list(self.token_count.items()):
            if token_count <= count:
                deleted_num += 1

                self.pop(token)
                self.token_count.pop(token)

        print(f"Deleted {deleted_num} tokens.")
        
    class _Adder():
        """Functional class that can be subscripted
        in order to call the `get_idx` method
        of a `Word2i` object with the add_to_dict
        setting set to True.

        *syntactic sugar*
        """
        def __init__(self, word2i : W):
            """Initialises the object.

            Parameters
            ----------
            word2i : Word2i
                The word2i mapping that the _Adder
                should refer to. 
            """
            self._word2i : W = word2i
        
        @overload
        def __getitem__(self, token : str) -> int:
            ...

        @overload
        def __getitem__(self, token : List[str]) -> List[int]:
            ...

        @overload
        def __getitem__(self, token : List[List[str]]) -> List[List[int]]:
            ...

        def __getitem__(self, token : Union[str, List[str], List[List[str]]]) \
                                -> Union[int, List[int], List[List[int]]]:
            """Converts token(s) to indices.
            Adds unseen tokens to the mapping.

            Parameters
            ----------
            token : Union[str, List[str], List[List[str]]]
                A single token, a list of tokens or
                a list of lists of tokens.

            Returns
            -------
            Union[int, List[int], List[List[int]]]
                The corresponding (list (of lists) of) index/indices.
            """
            return self._word2i.get_idx(token, add_to_dict = True)
    
    @property
    def add(self) -> _Adder:
        """A functional object that can be subscripted
        in order to call the `get_idx` method
        of the `Word2i` object with the add_to_dict
        setting set to True

        Returns
        -------
        _Adder
            The custom `_Adder`.
        """
        return self._Adder(self)
    
    def save(self, file_name : str, delimiter : str = ' ') -> None:
        """Writes the mapping's content
        into a file.

        Parameters
        ----------
        file_name : str
            Path including filename.
        delimiter : str, defaults to ' '
            Delimiter to use in file to
            separate token and index.
        """
        with open(file_name, "w") as file:
            assert(self.default_factory is not None)

            file.write(f"default{delimiter}{self.default_factory()}\n")

            for word, idx in self.items():
                file.write(f"{word}{delimiter}{idx}\n")

    @classmethod
    def load(cls, file_name : str, delimiter : str = ' ') -> Self:
        """Read a previously saved word2i
        mapping from a file. The file should
        contain a token and an index on
        each line.

        Parameters
        ----------
        file_name : str
            Path including filename.
        delimiter : str
            Character(s) that separate
            tokens from indices.

        Returns
        -------
        Self
            The loaded `Word2i` object.
        """
        with open(file_name, "r") as file:
            file_iter : Iterator[str] = iter(file)

            # load the default index
            default_int : int = int(next(file_iter).split(delimiter)[1])

            # load all other indices
            mapping : Dict[str, int] = {}

            split_line : List[str]
            token : str
            token_int : int
            for line in file_iter:
                split_line = line.split(delimiter)

                token = split_line[0]
                token_int = int(split_line[1])

                mapping[token] = token_int

            return cls(_mapping = mapping, _default = default_int)


class FutureWord2i(Word2i):
    """Extension to `Word2i` mapping that supports
    special tokens like bos, eos, sep and future.
    
    Converts tokens to indices.
    Acts as defaultdict, assigning a special
    unk_token to tokens that are not in the
    word2i dictionary.
    """
    def __init__(self, segmenter : Optional[Segmenter] = None,
                 init_add : bool = False,
                 bos_token : str = BOS_TOKEN,
                 eos_token : str = EOS_TOKEN,
                 sep_token : str = SEP_TOKEN,
                 pad_token : str = PAD_TOKEN,
                 cls_token : str = CLS_TOKEN,
                 mask_token : str = MASK_TOKEN,
                 future_token : str = FUTURE_TOKEN,
                 _mapping : Optional[Dict[str, int]] = None,
                 _default : int = 0) -> None:
        """Initialises a mapping from tokens to indices.
        If a segmenter is provided, it takes its `unk_token`
        index as the default index for the mapping and adds
        its special tokens directly to the mapping.
        If it is not provided, the special tokens given 
        as parameters are added to the mapping.

        Parameters
        ----------
        segmenter : Optional[Segmenter], defaults to None
            Segmenter to retrieve indices for tokens from.
            If None, indices are assigned incrementally.
        init_add : bool, defaults to False
            If True: takes the vocabulary from the tokenizer
            (if provided) at initialisation. If False:
            takes goal indices from tokenizer and adds them
            to the vocabulary manually when calling `.add[token]`.
        bos_token : str
        eos_token : str
        unk_token : str
        sep_token : str
        pad_token : str
        cls_token : str
        mask_token : str
        future_token : str
        """
        super().__init__(segmenter, init_add,
                         _mapping, _default)
        
        self.special_tokens : List[str]

        if segmenter is None:
            self.special_tokens = [bos_token, eos_token, sep_token,
                                   pad_token, cls_token, mask_token,
                                   future_token]
        
        else:
            self.special_tokens = [segmenter.bos_token,
                                   segmenter.eos_token,
                                   segmenter.sep_token,
                                   segmenter.pad_token,
                                   segmenter.cls_token,
                                   segmenter.mask_token,
                                   segmenter.future_token]
            
        for token in self.special_tokens:
            self.add[token]

    @override
    def del_rare(self, count : int) -> None:
        """Deletes tokens that have been seen
        only `count` times or less during 
        initialisation. Thus, they are assigned
        the unknown token from now on.

        Parameters
        ----------
        count : int
            Tokens seen this many times or less
            are replaced with unknown from now on.
        """
        
        deleted_num : int = 0

        for token, token_count in list(self.token_count.items()):
            if token_count <= count:
                if token not in self.special_tokens:
                    deleted_num += 1
    
                    self.pop(token)
                    self.token_count.pop(token)

        print(f"Deleted {deleted_num} tokens.")
        

class Char2i(Word2i):
    """Extension of `Word2i` class dedicated
    to character-level representations. When
    using the methods defined by its parent
    class, it ensures that each string only
    contains one character. Comes with additional
    methods that take care of automatic
    splitting of stings into their component
    characters."""
    def __init__(self,
                 _mapping : Optional[Dict[str, int]] = None,
                 _default : int = 0) -> None:
        """Initialises empty `Char2i` mappping.
        """
        super().__init__(_mapping = _mapping, _default = _default)  

        self._char_splitter = self._CharSplitter(self)

    @overload
    def __getitem__(self, token : str) -> int:
        ...

    @overload
    def __getitem__(self, token : List[str]) -> List[int]:
        ...

    @overload
    def __getitem__(self, token : List[List[str]]) -> List[List[int]]:
        ...

    @overload
    def __getitem__(self, token : List[List[List[str]]]) -> List[List[List[int]]]:
        ...

    @override
    def __getitem__(self, token : Union[str, List[str], List[List[str]], List[List[List[str]]]]) \
                     -> Union[int, List[int], List[List[int]], List[List[List[int]]]]:
        """Converts characters(s) to index/indices.
        Does not alter the underlying mapping.

        The input string(s) must comprise only
        of single characters.

        Parameters
        ----------
        token : Union[str, List[str], List[List[str]], List[List[List[str]]]]
            A single character, a list of characters,
            a list of lists of characters or
            a list of lists of lists of characters.

        Returns
        -------
        Union[int, List[int], List[List[int]]]
            The corresponding list(s) of indices
        """
        return self.get_idx(token, add_to_dict = False)
            
    @overload
    def get_idx(self, token : str, add_to_dict : bool = False) -> int:
        ...

    @overload
    def get_idx(self, token : List[str], add_to_dict : bool = False) -> List[int]:
        ...

    @overload
    def get_idx(self, token : List[List[str]], add_to_dict : bool = False) -> List[List[int]]:
        ...

    @overload
    def get_idx(self, token : List[List[List[str]]], add_to_dict : bool = False) -> List[List[List[int]]]:
        ...

    @override
    def get_idx(self, token : Union[str, List[str], List[List[str]], List[List[List[str]]]],
                    add_to_dict : bool = False) -> Union[int, List[int], List[List[int]], List[List[List[int]]]]:
        """Converts character(s) to indices.
        Can be used to add unseen tokens
        to the mapping.

        The input string(s) must comprise only
        of single characters.

        Parameters
        ----------
        token : Union[str, List[str], List[List[str]], List[List[List[str]]]]
            A single character, a list of characters,
            a list of lists of characters or
            a list of lists of lists of characters.
        add_to_dict : bool, defaults to False
            Whether to add characters to the
            mapping if they are not there already.

        Returns
        -------
        Union[int, List[int], List[List[int]], List[List[List[int]]]]
            The corresponding list(s) of indices
        """
        
        if isinstance(token, str):
            return self._retrieve(token, add_to_dict)
        
        else:
            out_list : List[int] | List[List[int]] | List[List[List[int]]] = []
            for t in token:
                out_list.append(super(self.__class__, self).get_idx(t, add_to_dict)) # type: ignore

            return out_list
    
    @override
    def _retrieve(self, token : str, add_to_dict : bool) -> int:
        """Converts a single character to its index.

        Parameters
        ----------
        token : str
            A single character.
        add_to_dict : bool, defaults to False
            Whether to add the character to the
            mapping if it is not in it already.

        Returns
        -------
        int
            The corresponding index.
        """
        assert len(token) == 1, "Char2i can only map chars (string of length 1) to indices."

        return super()._retrieve(token, add_to_dict)
    
    @property
    def chars(self) ->"_CharSplitter":
        """A functional object that can be subscripted
        with (lists of) words that contain a variable
        number of characters. The object takes care of
        splitting the words into character sequences and
        converts them using the `get_idx` method.

        Returns
        -------
        _CharSplitter
            The custom `_CharSplitter`.
        """
        return self._char_splitter
        
    class _CharSplitter():
        """Functional class that can be subscripted
        with (lists of) words that contain a variable
        number of characters. The object takes care of
        splitting the words into character sequences and
        converts them using the `get_idx` method.

        *syntactic sugar*
        """
        def __init__(self, char2i : C):
            """Initialises the object.

            Parameters
            ----------
            char2i : Char2i
                The char2i mapping that the _CharSplitter
                should refer to. 
            """
            self._char2i : C = char2i

            self._word2charids : Dict[str, List[int]] = {}
        
        @overload
        def get_idx(self, token : str, add_to_dict : bool = False) -> List[int]:
            ...

        @overload
        def get_idx(self, token : List[str], add_to_dict : bool = False) -> List[List[int]]:
            ...

        @overload
        def get_idx(self, token : List[List[str]], add_to_dict : bool = False) -> List[List[List[int]]]:
            ...

        def get_idx(self, token : Union[str, List[str], List[List[str]]], add_to_dict : bool = False) \
                                -> Union[List[int], List[List[int]], List[List[List[int]]]]:
            """Converts token(s) to lists of
            character indices.

            Parameters
            ----------
            token : Union[str, List[str], List[List[str]]]
                A single token, a list of tokens or
                a list of lists of tokens.
            add_to_dict : bool, defaults to False
                Whether to add unknown characters
                to dictionary.

            Returns
            -------
            Union[List[int], List[List[int]], List[int]]
                The corresponding list(s) of character
                indices.
            """

            if isinstance(token, str):
                try:
                    return self._word2charids[token]
                
                except KeyError:
                    characters : List[str] = self.split_characters(token)
                    ids : List[int] = self._char2i.get_idx(characters, add_to_dict = add_to_dict)
                    self._word2charids[token] = ids
                    return ids
            else:
                return [self.get_idx(word, add_to_dict) for word in token] #type: ignore
        
        @overload
        def __getitem__(self, token : str) -> List[int]:
            ...

        @overload
        def __getitem__(self, token : List[str]) -> List[List[int]]:
            ...

        @overload
        def __getitem__(self, token : List[List[str]]) -> List[List[List[int]]]:
            ...

        def __getitem__(self, token : Union[str, List[str], List[List[str]]]) \
                                -> Union[List[int], List[List[int]], List[List[List[int]]]]:
            """Converts token(s) to lists of
            character indices. Does not alter
            the underlying mapping.

            Parameters
            ----------
            token : Union[str, List[str], List[List[str]]]
                A single token, a list of tokens or
                a list of lists of tokens.

            Returns
            -------
            Union[List[int], List[List[int]], List[int]]
                The corresponding list(s) of character
                indices.
            """
            return self.get_idx(token, add_to_dict = False)
        
        @classmethod
        @overload
        def split_characters(cls, token : str) -> List[str]:
            ...
        
        @classmethod
        @overload
        def split_characters(cls, token : List[str]) -> List[List[str]]:
            ...

        @classmethod
        @overload
        def split_characters(cls, token : List[List[str]]) -> List[List[List[str]]]:
            ...
        
        @classmethod
        def split_characters(cls, token : Union[str, List[str], List[List[str]]]) \
                                -> Union[List[str], List[List[str]], List[List[List[str]]]]:
            """Splits a token or lists of tokens
            into individual characters.
            
            Parameters
            ----------
            token : Union[str, List[str], List[List[str]]]
                A single token, a list of tokens or
                a list of lists of tokens.

            Returns
            -------
            Union[List[str], List[List[str]], List[str]]
                The corresponding list(s) of characters.
            """
        
            if isinstance(token, str):
                return list(token)

            else:
                return [cls.split_characters(t) for t in token] # type: ignore
            
            #elif len(token) == 0 or isinstance(token[0], str):
            #    return [list(t) for t in token]
            #
            #else:
            #    return [[list(t) for t in sentence] for sentence in token]

        @property
        def add(self) -> "_CharAdder":
            """A functional object that can be subscripted
            with (lists of) words that contain a variable
            number of characters. The object takes care of
            splitting the words into character sequences,
            converts them using the `get_idx` method and
            adds unknown characters to the `Char2i` mapping.

            Returns
            -------
            _CharAdder
                The custom `_CharAdder`.
            """
            return self._CharAdder(self)

        class _CharAdder():
            """Functional class that can be subscripted
            with (lists of) words that contain a variable
            number of characters. The object takes care of
            splitting the words into character sequences,
            converts them using the `get_idx` method and
            calls the `get_idx` method of a `Char2i` object
            with the add_to_dict setting set to True.
            """
            def __init__(self, char_splitter):
                """Initialises the object.

                Parameters
                ----------
                char_splitter : _CharSplitter
                    The `_CharSplitter` to use for
                    splitting tokens into characters.
                """
                self._char_splitter = char_splitter

            @overload
            def __getitem__(self, token : str) -> List[int]:
                ...

            @overload
            def __getitem__(self, token : List[str]) -> List[List[int]]:
                ...

            @overload
            def __getitem__(self, token : List[List[str]]) -> List[List[List[int]]]:
                ...

            def __getitem__(self, token : Union[str, List[str], List[List[str]]]) \
                                    -> Union[List[int], List[List[int]], List[List[List[int]]]]:
                """Converts token(s) to character indices.
                Adds unseen characters to the mapping.

                Parameters
                ----------
                token : Union[str, List[str], List[List[str]]]
                    A single token, a list of tokens or
                    a list of lists of tokens.

                Returns
                -------
                Union[List[int], List[List[int]], List[List[List[int]]]]
                    The corresponding list(s) of character indices.
                """
                return self._char_splitter.get_idx(token, add_to_dict = True)


## DO NOT USE
#class SegmenterReader(Reader[TRP]):
#    def __init__(self, reader : Reader[RP], 
#                 tokenizer : _Segmenter[RP, TRP], 
#                 function : Callable[[RP], RP] = lambda rep : rep,
#                 length_limit : Optional[int] = None,
#                 include_start_end_markers : bool = False) -> None:
#        self.reader : Reader[RP] = reader
#        self.tokenizer : _Segmenter[RP, TRP] = tokenizer
#        self.function : Callable[[RP], RP] = function
#        self.length_limit : Optional[int] = length_limit
#        self.include_start_end_markers : bool = include_start_end_markers
#    
#    def __iter__(self) -> Iterator[TRP]:
#        for representation in self.reader:
#            tokenized : TRP = self.tokenizer(self.function(representation),
#                                             as_sentence = self.include_start_end_markers)
#            if self.length_limit is not None:
#                if len(tokenized.tokens) <= self.length_limit:
#                    yield tokenized
#            else:
#                yield tokenized
#
#class FlatSegmenterReader(SegmenterReader[TokenizedRepresentation]):
#    def __init__(self, reader : FlatReader, 
#                 tokenizer : _Segmenter[Representation, TokenizedRepresentation],
#                 length_limit : Optional[int] = None,
#                 include_start_end_markers : bool = False):
#        super().__init__(reader, tokenizer, length_limit = length_limit)
#        self.include_start_end_markers : bool = include_start_end_markers
#
#class PTBTreeSegmenterReader(SegmenterReader[PTBTreeRepresentation]):
#    def __init__(self, reader : PTBReader, 
#                 tokenizer : TreeSegmenter, 
#                 length_limit : Optional[int] = None,
#                 detokenize : bool = True):
#        function : Callable[[PTBTreeRepresentation], PTBTreeRepresentation]
#        if detokenize:
#            function = lambda ptbtree : ptbtree.detokenize(inplace = False)
#        else:
#            function = lambda ptbtree : ptbtree
#
#        super().__init__(reader, tokenizer, function, length_limit, False)
#