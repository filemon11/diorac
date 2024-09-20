"""A collection of classes that represent sentences
and their underlying syntax.

Classes
----------
Tree
    NLTK Tree extension that keeps track of leaf indices.
Representation
    Represents simple sentence.
TokenizedRepresentation
    Abstract representation class prescribing methods for tokenized representations.
FlatTokenizedRepresentation
    Representation that features sentence separation.
TreeRepresentation
    Representation that features an underlying syntactic tree.
PTBTreeRepresentation
    Extension of ``PTBTreeRepresentation`` that features several PTB specific transformations.

Methods
----------
write_trees(trees, file_path)
    Writes trees to file.
"""

from nltk import Tree as NltkTree
from nltk.tokenize.treebank import TreebankWordDetokenizer

import re

from abc import ABC, abstractproperty, abstractmethod

from typing import List, Union, Tuple, Optional, TypeVar, Sequence, Iterable, Literal, Dict
from typing import overload
from typing_extensions import Self

WORD_TAGS = (
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
    'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
    'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
    'WP$', 'WRB',
)
"""PTB POS tags that mark words."""

n_p = ('n\'t', '\'s', '\'m', '\'re', '\'ve', '\'d', '\'ll', '%', "'")
NEG_POSS = n_p + tuple(p.upper() for p in n_p)
"""The tuple of clitics which are already tokenized in the PTB."""

TRAILING_PUNCT_TAGS = ('.', ',', ':', 'RRB', 'RCB', '\'\'', '...')
"""PTB POS tags for punctuation that is appended to a word."""

INIT_PUNCT_TAGS = ('``', '`', 'LRB', 'LCB', '$', '#', "''")
"""PTB POS tags for punctuation that is prepended to a word."""

PUNCT_TAGS = TRAILING_PUNCT_TAGS + INIT_PUNCT_TAGS
"""PTB POS tags for punctuation symbols."""

TRAILING_PUNCT_SYM = ('.', ',', ':', 'RRB', 'RCB', '\'\'', '...', '"', '?', '!', "'", ")")
"""Punctuation that is appended to a word."""

INIT_PUNCT_SYM = ('``', '`', 'LRB', 'LCB', '$', "''", "'", "(")
"""Punctuation that is prepended to a word."""


RP = TypeVar("RP", bound = "Representation")
T = TypeVar("T", bound = "Tree")

class Tree(NltkTree):
    """``nltk.Tree`` extension that keeps track of leaf indices.

    Leaves are represented as a str parented
    by a Tree with a POS-tag as label. The `start_idx`
    attribute can be used to match a transformed
    tree with its source tree.

    Attributes
    ----------
    start_idx : int
        Index of the node's leftmost leaf.

    """

    def __init__(self, label : str, children : Iterable[Union[str, "Tree"]] = [], start_idx : Optional[int] = None):
        """Initialises a Tree object.

        Parameters
        ----------
        label : str
            Label of the node. 
        children : Iterable[str | Tree], defaults to []
            Child nodes.
        start_idx : int, optional
            Index of the node's leftmost leaf.
            If `None`, then `Tree.start_idx` 
            is initialised with the minimum start_idx
            of the items in the children parameter. If it
            contains no `Tree`, `Tree.start_idx` is
            initialised with 0.

        """
        super().__init__(label, children)

        self.start_idx : int
        """Minimum index of the node's leaves."""

        if start_idx is None:
            children_start_idxs : List[int]
            children_start_idxs = [child.start_idx for child 
                                   in children if not isinstance(child, str)]

            # if there are no non-string children, i.e. if node is leaf
            if len(children_start_idxs) == 0:
                self.start_idx = 0
            
            # if node is not a leaf
            else:
                self.start_idx = min(children_start_idxs)

        else:
            self.start_idx = start_idx


    @overload
    def lower(self, inplace : Literal[True]) -> None:
        ...

    @overload
    def lower(self, inplace : Literal[False] = False) -> Self:
        ...

    def lower(self, inplace : bool = False) -> Optional[Self]:
        """Turns all leaves into
        lowercase.

        Parameters
        ----------
        inplace : bool, defaults to False
            Whether to perform the operation
            inplace or make a copy of the tree
            and perform the operation there

        Returns
        -------
        Optional[Tree]
            The new tree if `inplace = False`
        """

        def _lower(node : Self) -> None:
            """Recursively turns a node
            into lowercase.
            """
            for i, child in enumerate(node):
                if isinstance(child, str):
                    node[i] = child.lower()
                else:
                    _lower(child)

        tree_to_lower : Self = self
        if inplace:
            tree_to_lower = self.deepcopy()
        _lower(tree_to_lower)

        return tree_to_lower

    def reindex(self, start_idx : Literal[0, 1] = 0) -> None:
        """Recursively reassign indices to the node and all
        of its descendants. Starts numbering with 
        the leftmost leaf.

        Parameters
        ----------
        start_idx : Literal[0, 1], defaults to 0
            Index to start numbering with.

        """

        def _reindex(tree : Self, start_idx : int = 0) -> int:
            """Recursively reassign indices to the node and all
            of its descendants. Starts numbering with 
            the leftmost leaf.

            Parameters
            ----------
            start_idx : int, defaults to 0
                Index to start numbering with.

            Returns
            ----------
            int
                Index of the node's rightmost leaf.
            """
            tree.start_idx = start_idx

            # if it is not a leaf, go through all children
            if not (len(tree) == 1 and isinstance(tree[0], str)):

                for child in tree:
                    assert isinstance(child, Tree), "Node has both string and Node as daughters."
                    start_idx = _reindex(child, start_idx)
                    start_idx += 1
                else:
                    start_idx -= 1  # subtract at the end, since no child follows.

            return start_idx
        
        _reindex(self, start_idx)
    
    @overload
    def remove_traces(self, inplace : Literal[True] = True) -> None:
        ...

    @overload
    def remove_traces(self, inplace : Literal[False]) -> Self:
        ...

    def remove_traces(self, inplace : bool = False) -> Optional[Self]:
        """Remove trace nodes. All words that begin with *
        or have -NONE- as label
        are treated as traces.
        
        Parameters
        ----------
        inplace : bool, defaults to False
            Whether to perform the operation
            inplace or make a copy of the tree
            and perform the operation there

        Returns
        -------
        Optional[Tree]
            The new tree if `inplace = False`
        """

        def _remove_traces(node : Union[Self, str]) -> bool:
            """Recursively removes all 
            subtrees that only contain traces.

            Parameters
            ----------
            node : Self | str
                The node to modify.
    
            Returns
            -------
            bool
                Whether `node` is a tree
                that only contains traces
                or is empty.
            """

            if isinstance(node, str):
                if node.startswith("*"):
                    return True
                else:
                    return False
            
            elif node.label() == "-NONE-":
                return True
                
            else:
                for i, child in reversed(list(enumerate(node))):
                    if _remove_traces(child):
                        node.pop(i)

                if len(node) == 0:
                    return True
                else:
                    return False

        if inplace:
            _remove_traces(self)
            return None
        
        else:
            copied_tree : Self = self.deepcopy()
            _remove_traces(copied_tree)
            return copied_tree

    @classmethod
    def wrap(cls, nltktree : NltkTree, start_idx : Literal[0, 1] = 0) -> NltkTree:
        """Recursively converts an nltk.Tree into 
        a Tree by assigning indices.

        Parameters
        ----------
        nltktree : nltk.Tree
            Tree to convert.
        start_idx : Literal[0, 1], defaults to 0
            Index to start numbering with.
        
        Returns
        ----------
        converted_tree : Tree
            The converted tree.

        """

        @overload
        def _wrap(nltktree : str, start_idx : int) -> Tuple[str, int]:
            ...

        @overload
        def _wrap(nltktree : NltkTree, start_idx : int) -> Tuple[NltkTree, int]:
            ...

        def _wrap(nltktree : Union[str, NltkTree], start_idx : int = 0) -> Tuple[Union[str, Self], int]:
            """Recursively converts an nltk.Tree into 
            a Tree by assigning indices.

            Parameters
            ----------
            nltktree : str | nltk.Tree
                Tree to convert.
            start_idx : int, defaults to 0
                Index to start numbering with.

            Returns
            ----------
            converted_tree : str | Tree
                The converted tree.
            end_idx : int
                Index of the tree's rightmost leaf.

            """
            # if it is a string, there is no need for transformation
            if isinstance(nltktree, str):
                return nltktree, start_idx

            # Create a new tree node
            new_tree : Self = cls(nltktree.label(), [], start_idx)

            # add children iteratively
            new_child : Union[str, Tree]
            for child in nltktree:
                new_child, start_idx = _wrap(child, start_idx) # create daughters recursively
                start_idx += 1
                new_tree.append(new_child)

            else:
                start_idx -= 1  # subtract at the end, since no child follows.

            return new_tree, start_idx
    
        return _wrap(nltktree, start_idx)[0]

    @classmethod
    def fromstring(cls, line : str, brackets : str = "()") -> Self:
        """Parse a tree from a string.

        Parameters
        ----------
        line : str
            String containing the tree.
        brackets : str, defaults to '()'
            Bracket format to use. Must be
            two-character string.
        
        Returns
        ----------
        Tree
            The parsed tree.

        Raises
        ------
        AssertionError
            If string only contains a leaf with no POS tags.

        """
        assert len(brackets) == 2, "brackets argument must be a two-character string."

        wrapped : Union[str, NltkTree] = cls.wrap(super().fromstring(line, brackets = brackets))

        assert isinstance(wrapped, NltkTree), "Tree contains only leaf and no POS tags."
        return wrapped

    def deepcopy(self) -> Self:
        """Creates a copy of the node
        and all descendant nodes.

        Returns
        ----------
        Tree
            The copy tree.

        """
        children : List[Self | str] = list(self)
        return self.__class__(self.label(), 
                    [(child if isinstance(child, str) 
                            else child.deepcopy()) for child in children], 
                    self.start_idx)
    
    def __repr__(self) -> str:
        """Create representation
        for Tree.

        Returns
        ----------
        str
            The representation.

        """
        return str(self)
    
    def str_representation(self, brackets : str = "[]",
                           include_idx : bool = False) -> str:
        """Convert string to
        one-line bracket format.

        Parameters
        ----------
        brackets : str, defaults to '[]'
            Bracket symbols to use.
        include_idx : bool, defaults to False
            Whether to include indices.

        Returns
        ----------
        str
            The string representation.

        """
        assert len(brackets) == 2, (f"brackets argument does "
                                    f"not contain exactly two characters: {str(brackets)}")

        def _as_string(node : Self | str,
                       brackets : str,
                       include_idx : bool) -> str:
            """Give string representation of node.
            Returns the node if it is a string,
            else gives node.str_representation.
            """
            if isinstance(node, str):
                return node
            else:
                return node.str_representation(brackets, include_idx)
            
        return (f"{brackets[0]}{self.label()}{':'+str(self.start_idx) if include_idx else ''} "
                f"{' '.join([_as_string(child, brackets, include_idx) for child in self])}{brackets[1]}")
    
    def __str__(self) -> str:
        """Convert string to
        one-line bracket format.

        Parameters
        ----------
        brackets : str, defaults to '[]'
            Bracket symbols to use.

        Returns
        ----------
        str
            The string representation.

        """
        return self.str_representation()

class Representation(str):
    """Represents simple sentence.
    """

    punctuation_re : str = r"(?<!\d)[.,;:?!](?!\d)"
    """Regex to remove punctuation symbols except between digits."""

    @classmethod
    def _remove_punctuation(cls, string : str) -> str:
        """Removes punctuation from a string.

        Parameters
        ----------
        string : str
            A string.

        Returns
        ----------
        str
            The string without punctuation.
        """
        return re.sub(cls.punctuation_re, "", string, 0)
    
    def remove_punctuation(self) -> Self:
        """Creates a new Representation without punctuation.
        
        Returns
        ----------
        Self
            The representation.
        """
        return self.__class__(self._remove_punctuation(self))

class TokenizedRepresentation(Representation, ABC):
    """Abstract representation class prescribing methods 
    for tokenized representations.
    """

    _special_sym_rem_tranl_tab : dict[int, int | None] = str.maketrans("", "", "Ġ")
    """Translation table for the removal of special symbols like Ġ."""

    @abstractproperty
    def tokens(self) -> List[str]:
        """The tokens of the representation."""
        ...

    @classmethod
    def _remove_special_symbol(cls, token : str) -> str:
        """Remove sepcial space
        symbol from token.
        
        Parameters
        ----------
        token : str
            The token.

        Returns
        -------
        str
            The altered token.
        """
        return token.translate(cls._special_sym_rem_tranl_tab)
    
    @abstractmethod
    def limit_to(self, position : int, retain_last : bool) -> Self:
        """Creates a new representation
        that includes every token up to
        the position given as parameter
        (not including it).

        Parameters
        ----------
        position : int
            Position up to which tokens
            should be included.
        retain_last : bool
            Whether to retain the last
            token.

        Returns
        -------
        Self
            A new representation fragment.
        """
        ...


class FlatTokenizedRepresentation(TokenizedRepresentation):
    """Extension to a simple string representation
    that allows to store an underlying tokenization.

    Attributes
    ----------
    indices : List[int]
        A list of the tokens' indices. Can be used to keep
        track of transformations.

    """
    def __new__(cls, string : Optional[str], 
                tokens : List[str], 
                indices : Optional[List[int]] = None) -> Self:
        """Create new `FlatTokenizedRepresentation`.

        Parameters
        ----------
        string : str | None
            The string representation. If None, it
            is generated by concatenating the tokens.
        tokens : List[str]
            The underlying tokens.
        indices : List[int], optional
            A list of indices. If None, they
            are generated based on the token length,
            starting with 0.

        Returns
        -------
        FlatTokenizedRepresentation
            The object.
        """
        if string is None:
            string =  cls.make_string(tokens)
            
        return TokenizedRepresentation.__new__(cls, string)
    
    def __init__(self, string : Optional[str], tokens : Iterable[str], indices : Optional[Iterable[int]] = None) -> None:
        """Initialise new `FlatTokenizedRepresentation`.

        Parameters
        ----------
        string : str | None
            The string representation. If None, it
            is generated by concatenating the tokens.
        tokens : List[str]
            The underlying tokens.
        indices : List[int], optional
            A list of indices. Must have the same length
            as tokens. If None, the indices are generated 
            based on the token length, starting with 0.

        """

        self._tokens : List[str] = list(tokens)
        """Internal: List of the tokens."""
        
        self.indices : List[int]
        """List of indices."""

        if indices is None:
            self.indices = list(range(len(self._tokens)))
        else:
            self.indices = list(indices)

        # Check whether token and indice lists match in length
        assert(len(self._tokens) == len(self.indices))

    def limit_to(self, position: int, retain_last : bool) -> Self:
        """Creates a new representation
        that includes every token up to
        the position given as parameter
        (not including it).

        Parameters
        ----------
        position : int
            Position up to which tokens
            should be included.
        retain_last : bool
            Whether to retain the last token.

        Returns
        -------
        Self
            A new representation fragment.
        """
        new_tokens : List[str] = self.tokens[:position]
        new_indices : List[int] = self.indices[:position]

        if retain_last:
            new_tokens.append(self.tokens[-1])
            new_indices.append(self.indices[-1])
        
        return self.__class__(string = None,
                              tokens = new_tokens,
                              indices = new_indices)

    @classmethod
    def make_string(cls, tokens : List[str], delimiter = ' ') -> str:
        """Create string from list of tokens.
        
        Parameters
        ----------
        string : List[str]
            Tokens.
        delimiter : str, defaults to ' '
            Character(s) to insert between
            concatenated tokens.

        Returns
        ----------
        str
            The concatenated string.
        """
        return delimiter.join([cls._remove_special_symbol(token) for token in tokens])

    def remove_punctuation(self) -> Self:
        """Creates a new Representation without punctuation.
        
        Returns
        ----------
        Self
            The representation.
        """

        tokens_wo_punct : List[str] = [self._remove_punctuation(token) for token in self.tokens]

        tokens_updated : List[str] = []
        indices_updated : List[int] = []

        for token, index in zip(tokens_wo_punct, self.indices):
            if len(self._remove_special_symbol(token)) != 0:
                tokens_updated.append(token)
                indices_updated.append(index)

        return self.__class__(string = None,
                              tokens = tokens_updated,
                              indices = indices_updated)

    def lower(self) -> Self:
        """Generate lowercase version
        of the representation (including its tokens).

        Returns
        -------
        Self
            The new lowercase representation.
        """

        return self.__class__(str(self).lower(), 
                              [t.lower() for t in self.tokens], 
                              self.indices)
    @property
    def tokens(self) -> List[str]:
        """The tokens of the representation."""
        return self._tokens
    
    
class TreeRepresentation(TokenizedRepresentation):
    """A representation that stores
    an underlying tree.

    This is a tokenized representation since
    the creation of a tree implies some kind
    of tokenization into leaves.

    Attributes
    ----------
    tree : Tree
        The underlying tree.
    
    """
    def __new__(cls, string : Optional[str], tree : Tree) -> Self:
        """Create a `TreeRepresentation`.

        Parameters
        ----------
        string : str | None
            The string representation. If None,
            it retrieves the string from the tree 
            leaves by concatenation.
        tree : Tree
            The underlying tree.

        Returns
        -------
        TreeRepresentation
            The new object.
        """
        if string is None:
            string = cls.make_string(tree)

        return TokenizedRepresentation.__new__(cls, string)
    
    def __init__(self, string : Optional[str], tree : Tree):
        """Initialise a `TreeRepresentation`.

        Parameters
        ----------
        string : str | None
            The string representation. If None,
            it retrieves the string from the tree 
            leaves by concatenation.
        tree : Tree
            The underlying tree.
        """
        self.tree : Tree = tree
        """The underlying tree."""

    def limit_to(self, position: int, retain_last : bool) -> Self:
        """Creates a new representation
        that includes every leaf up to
        the position given as parameter
        (not including it). Simply removes
        all other leaves and resulting
        empty nodes.

        Parameters
        ----------
        position : int
            Position up to which tokens
            should be included.
        retain_last : bool
            Whether to retain
            the last leaf.

        Returns
        -------
        Self
            A new representation fragment.
        """
        #TODO
        def get_leaves(tree : Tree) -> List[Tree]:
            if len(tree) == 1 and isinstance(tree[0], str):
                return [tree]
            else:
                return [leaf for child in tree for leaf in get_leaves(child)]
        
        new_tree : Tree = self.tree.deepcopy()
        leaves : List[Tree] = get_leaves(new_tree)

        for i, leaf in reversed(list(enumerate(leaves))):
            if retain_last and i == len(leaves)-1:
                continue
            
            leaf.clear()
            if i == position:
                break
        
        self._remove_empty_nodes(new_tree)

        return self.__class__(string = None, tree = new_tree)

    def remove_punctuation(self, inplace : bool = False) -> Self:
        """Creates a new Representation without punctuation.
        
        Parameters
        ----------
        inplace : bool, defaults to False
            Whether to perform some of the
            tree altering operations inplace to
            speed up the process.

        Returns
        ----------
        Self
            The representation.
        """

        def _remove_punct_leaves(tree : Tree) -> bool:
            """Recursively remove punctuation leaves from
            tree. Performs inplace-action.
            
            Parameters
            ----------
            tree : Tree
                Tree node.

            Returns
            ----------
            bool
                Whether to remove the node.
                
            """
            child_wo_punct : str
            for i, child in sorted(enumerate(tree), reverse = True, key = lambda x : x[0]):

                if isinstance(child, str):
                    child_wo_punct = self._remove_punctuation(child)

                    if len(self._remove_special_symbol(child_wo_punct)) == 0:
                        tree.pop(i)

                    else:
                        tree[i] = child_wo_punct

                else:
                    if _remove_punct_leaves(child):
                        tree.pop(i)
            
            if len(tree) == 0:
                return True
            
            else:
                return False

        tree_to_alter : Tree = self.tree
        if not inplace:
            tree_to_alter = tree_to_alter.deepcopy()

        _remove_punct_leaves(tree_to_alter)

        return self.__class__(string = None,
                              tree = tree_to_alter)

    def lower(self) -> Self:
        """Generate a lowercase version
        of thre representation, including
        its tree leaves. 

        Returns
        -------
        Self
            The new representation
        """

        return self.__class__(super().lower(), 
                              self.tree.lower(inplace = False))

    @classmethod
    def make_string(cls, tree : Tree, delimiter : str = ' ') -> str:
        """Create a string from a tree
        by concatenating its leaves.

        Parameters
        ----------
        tree : Tree
            The tree for string extraction.
        delimiter : str, defaults to ' '
            The character(s) to insert
            in between individual tokens.

        Returns
        -------
        str
            The generated string.
        """
        return delimiter.join(tree.leaves())
    
    @property
    def leaves(self) -> List[str]:
        """The tree leaves."""
        return self.tree.leaves()

    @property
    def tokens(self) -> List[str]:
        """The tree tokens, i.e. its leaves."""
        return self.leaves
    
    def deepcopy(self) -> Self:
        """Create a deep copy of the representation.
        
        Returns
        -------
        TreeRepresentation
            The deep copy.
        """
        return self.__class__(self, self.tree.deepcopy())
    
    @classmethod
    def from_string(cls, string : str, brackets : str = "()") -> Self:
        """Creates a tree representation
        from one-line bracketed string format.

        Parameters
        ----------
        string : str
            The bracketed tree encoding.
        brackets : str, defaults to '()'
            The bracket format to use.
            Must be a two-character string.

        Returns
        -------
        Self
            The corresponding tree representation.
        """

        tree : Tree = Tree.fromstring(string, brackets)
        return cls(string = None, tree = tree)

    @classmethod  
    def _remove_empty_nodes(cls, tok_tree : T) -> Optional[T]:
        """Recursively Remove empty nodes 
        from a tree. Warning: Does not create 
        a copy of the tree.

        Parameters
        ----------
        tok_tree : Tree
            Tree to remove empty nodes from.

        Returns
        -------
        Tree | None
            Resulting tree. Is None if it is
            empty itself.
        """
        
        children : List[Union[T, str, None]]
        children = [(child if isinstance(child, str)
                     else cls._remove_empty_nodes(child)) for child in tok_tree]

        filtered_children : List[Union[T, str]]
        filtered_children = [child for child in children if child is not None]

        if len(filtered_children) == 0:
            return None
        
        else:
            tok_tree.clear()
            tok_tree.extend(filtered_children)
            return tok_tree

    
class PTBTreeRepresentation(TreeRepresentation):
    """Extension of the `TreeRepresentation` that
    features a variety of methods to treat trees
    that arise from Penn Treebank (PTB) annotation.

    Attributes
    ----------
    detokenized : bool
        Whether the tree has been detokenized.

    """
    _detokenizer : TreebankWordDetokenizer = TreebankWordDetokenizer()
    """Internal: Utility to reverse the clitic tokenization that was
    performed in the PTB (e.g. can 't => can't)"""
    
    def __new__(cls, string : Optional[str], tree : Tree, detokenized : bool = False) -> Self:
        """Create a `TreeRepresentation`.

        Parameters
        ----------
        string : str | None
            The string representation. If None,
            it retrieves the string from the tree 
            leaves by concatenation.
        tree : Tree
            The underlying tree.
        detokenized : bool, defaults to False
            Whether the tree has been detokenized.

        Returns
        -------
        Self
            The new object.
        """
        return TreeRepresentation.__new__(cls, string, tree)
    
    def __init__(self, string : Optional[str], tree : Tree, detokenized : bool = False):
        """Initialise a `TreeRepresentation`.

        Parameters
        ----------
        string : str | None
            The string representation. If None,
            it retrieves the string from the tree 
            leaves by concatenation.
        tree : Tree
            The underlying tree.
        detokenized : bool, defaults to False
            Whether the tree has been detokenized.
        """
        super().__init__(string, tree)

        self.detokenized : bool = detokenized
        """Whether the tree has been detokenized."""

    def detokenize(self, delimiter : str = ' ', 
                   inplace : bool = True, 
                   remove_non_words : bool = True,
                   remove_traces : bool = True) -> Self:
        """Reverse the tokenization of clitics performed
        in the original Penn Treebank. Detokenizes both
        string and the underlying tree by merging nodes.

        Warning: `inplace = True` still returns a new
        object. See below for more information.

        Changes node labels: 
        - POS tags of merged leaves are
          combined with the symbol '+'.
        - If the merged leaf sat at the bottom
          of a unary chain, it is preserved by
          merging labels with '@'.

        Parameters
        ----------
        delimiter : str, defaults to ' '
            Delimiter to use.
        inplace : bool, defaults to True
            Whether to perform some of the transformations
            in the existing objects.
        remove_non_words : bool, defaults to True
            Whether to remove punctuation and traces.
            Warning: Equality of string and tree.leaves
            not tested for remove_non_words = False.
        remove_traces : bool, defaults to True
            If `remove_non_words == False`: Whether to
            remove trace nodes. Punctuation is
            retained.
        
        Returns
        -------
        PTBTreeRepresentation
            The detokenized representation.

        Raises
        -------
        TreeTransformError
            If there is an error in the transformation.
            Mainly occurs if the annotation is faulty.
        """
        assert not self.detokenized, "Tried to detokenize but the tree is already detokenized."

        #if not remove_non_words or not remove_traces:
        #    raise NotImplementedError("Please set remove_non_words to True. Not implemented yet!")
        #        # TODO: allow to remove traces but leave punctuation
        #        # TODO: allow opening quotation mark right merging

        words : List[str]
        tree_to_detok : Self = self if inplace else self.deepcopy()

        if remove_non_words:
            tree_to_detok = tree_to_detok.remove_non_words()    # Removes all non-words from the tree,
                                                                # as well as resulting empty nodes
            words = tree_to_detok.leaves

        elif remove_traces:
            tree_to_detok = tree_to_detok.remove_traces()
            words = tree_to_detok.leaves

        else:
            words = self.leaves # including punctuation and traces

        # detokenize the leaves
        string : str = self._detokenizer.detokenize(words)
        string_old = string
        string = self.double_check_detokenize(string, delimiter = delimiter)
    
        # detokenize the tree and transform it
        tree : Tree
        prev_word_node : Optional[Tree]   # sometimes a left punctuation mark is misplaced
                                # it should be appended to the last word leaf
        misplaced_node : Optional[Tree]

        tree, prev_word_node, misplaced_node = self._detok_tree_recursive(tree = tree_to_detok.tree,
                                                    merge_punct = not remove_non_words)
        
        if misplaced_node is not None:
            assert prev_word_node is not None, "There is not word node in the tree."
            self._merge_leaves(prev_word_node, misplaced_node, "right")

        
        tree_wo_empty_nodes : Optional[Tree] = self._remove_empty_nodes(tree)
        assert tree_wo_empty_nodes is not None, "Data contains empty tree. Please remove it."
        
        tree = tree_wo_empty_nodes
        
        if not string == delimiter.join(tree.leaves()):
            raise TreeTransformError("Error in non-word removal logic."
                                     " String and leaves do not match."
                                     " Maybe check for the completeness"
                                     " of NEG_POSS."
                                     f" '{string}' != '{delimiter.join(tree.leaves())}'\n"
                                     f" {str(self.tree)}\n"
                                     f" stringold: {string_old}")
        
        return self.__class__(string, tree, detokenized = True)
    
    @staticmethod
    def _merge_leaves(tree : T, tree_to_merge : Optional[T], direction : Literal["left", "right"]) -> None:
        """Merge two leaf nodes.
        
        Parameters
        ----------
        tree : Tree
            The tree that elements should be
            added to.
        tree_to_merge : Tree, optional
            Tree that should be added to
            the first parameter tree. If None,
            no action is performed.
        direction : Literal['left', 'right']
            Side that tree_to_merge
            should be added at.
        """
                
        #replace_apos = lambda w : '"' if w == '``' or w == "''" else w
        if tree_to_merge is not None:
        
            if direction == "left":
                tree.set_label(f"{tree_to_merge.label()}+{tree.label()}")
                tree[0] = _replace_punct(tree_to_merge[0]) + _replace_punct(tree[0])
            else:
                tree.set_label(f"{tree.label()}+{tree_to_merge.label()}")
                tree[0] = _replace_punct(tree[0]) + _replace_punct(tree_to_merge[0])

    @overload
    @classmethod
    def _detok_tree_recursive(cls, tree : T, 
                              prev_word_node : None = None,
                              label_chain : List[str] = [],
                              prev_left_punct : Optional[T] = None,
                              neg_poss : Sequence[str] = NEG_POSS,
                              word_tags : Sequence[str] = WORD_TAGS,
                              prec_punct_tags : Sequence[str] = INIT_PUNCT_TAGS,
                              succ_punct_tags : Sequence[str] = TRAILING_PUNCT_TAGS,
                              merge_punct : bool = False) -> Tuple[T, Optional[T], Optional[T]]:
        ...
    
    @overload
    @classmethod
    def _detok_tree_recursive(cls, tree : T, 
                              prev_word_node : T,
                              label_chain : List[str] = [],
                              prev_left_punct : Optional[T] = None,
                              neg_poss : Sequence[str] = NEG_POSS,
                              word_tags : Sequence[str] = WORD_TAGS,
                              prec_punct_tags : Sequence[str] = INIT_PUNCT_TAGS,
                              succ_punct_tags : Sequence[str] = TRAILING_PUNCT_TAGS,
                              merge_punct : bool = False) -> Tuple[Optional[T], T, Optional[T]]:
        ...
        
    @classmethod
    def _detok_tree_recursive(cls, tree : T, 
                              prev_word_node : Optional[T] = None,
                              label_chain : List[str] = [],
                              prev_left_punct : Optional[T] = None,
                              neg_poss : Sequence[str] = NEG_POSS,
                              word_tags : Sequence[str] = WORD_TAGS,
                              prec_punct_tags : Sequence[str] = INIT_PUNCT_TAGS,
                              succ_punct_tags : Sequence[str] = TRAILING_PUNCT_TAGS,
                              merge_punct : bool = False) -> Tuple[Optional[T], Optional[T], Optional[T]]:
        """Recursively detokenizes a `Tree` by
        merging leave nodes that were split because
        of the clitic tokenization performed in the PTB.
        
        Changes node labels: 
        - POS tags of merged leaves are
          combined with the symbol '+'.
        - If the merged leaf sat at the bottom
          of a unary chain, it is preserved by
          merging labels with '@'.

        
        Parameters
        ----------
        tree : Tree
            The tree to detokenize. It is altered
            in-place.
        prev_word_node : Tree, optional
            The preceding leaf node that a clitic
            should merge into. Skips traces.
        label_chain : List[str]
            All parent nodes up to the
            first antecedent that has more than one
            daughter; includes further antecedent nodes
            if the preceding cousins are all merged and
            this is the last cousin.
        prev_left_punct : Tree, optional
            Previous punctuation that should be
            prepended from the left to the next word.
        neg_poss : Sequence[str], defaults to NEG_POSS
            Sequence of clitics to merge.
        word_tags : Sequence[str], defaults to WORD_TAGS
            POS tags indicating words. All other leaf
            tags are skipped in merging operations.
        prec_punct_tags : Sequence[str], defaults to INIT_PUNCT_TAGS
            POS tags of preceding punctuation symbols.
        succ_punct_tags : Sequence[str], defaults to TRAILING_PUNCT_TAGS
            POS tags of succeeding punctuation symbols.
        #merge_punct : bool, defaults to False
        #    Set to True if you also want to merge punctuation
        #    symbols with preceding words.

        Returns
        ----------
        detokenized_node : Tree | None
            The detokenized node. Is None if it
            was merged with the preceding word node.
        prev_word_node : Tree | None
            The preceding word node. Is None if there
            was no word node in the tree up to this point.
        prepend_punct_node : Tree | None
            Punctuation node that should be prepended to the
            next word token.
        prev_leaf_node : Tree | None
            The preceding leaf node. Is None if there
            was no leaf node in the tree up to this point.
        """
        
        if len(tree) == 1 and isinstance(tree[0], str):
            # if the node represents a leaf, either merge it or leave it as is

            should_merge_clitic : bool = tree[0] in neg_poss
            should_merge_succ_punct : bool = (merge_punct and tree.label() in succ_punct_tags
                                              and tree[0] != "--")

            unary_chain : str

            if (should_merge_clitic or should_merge_succ_punct) and prev_word_node is not None \
                and tree[0] != "--" and tree[0] != "-":
                # TODO: Should this raise an Exception if prev_word_node is None?
                # only merge if a preceding word node exists


                unary_chain = '@'.join(label_chain + [tree.label()])
                prev_word_node.set_label(unary_chain)

                if prev_left_punct is None:

                    cls._merge_leaves(prev_word_node, tree, 'right')

                    return None, prev_word_node, prev_left_punct     # Returns None since the node was merged
                
                else:
                    cls._merge_leaves(prev_left_punct, tree, 'right')

                    return None, prev_word_node, prev_left_punct
            
            elif merge_punct and tree.label() in prec_punct_tags and tree[0] != "--" and tree[0] != "-":

                unary_chain = '@'.join(label_chain + [tree.label()])
                tree.set_label(unary_chain)

                cls._merge_leaves(tree, prev_left_punct, "left")  # Merges both left punctuation nodes


                return None, prev_word_node, tree
            
            else:
                # Do nothing if not to be merged               
                if tree.label() in word_tags:

                    cls._merge_leaves(tree, prev_left_punct, "left")

                    return tree, tree, None     
                                                # Return the tree as the new prev_word_tree
                                                # if it represents a word leaf.
                
                else:
                    return tree, prev_word_node, prev_left_punct

        else:
            new_children : List[T] = []     # initialise new list if children nodes to which
                                            # only the non-merged nodes are added

            updated_tree : Optional[T]

            new_label_chain : List[str] = []

            for i, child in enumerate(tree):
                if len(new_children) == 0 and i == len(tree) - 1:
                    # if the new node has no leaves, i.e. all previous leaves were merged
                    # and the current child is the last/rightmost one.
                    
                    new_label_chain = label_chain + [tree.label()]
                    # Forward the current and the ancestor labels so they do not get lost.

                # get detokenized child and update preceding word node
                updated_tree, prev_word_node, prev_left_punct = cls._detok_tree_recursive(child, 
                                                                                          prev_word_node, 
                                                                                          new_label_chain,
                                                                                          prev_left_punct,
                                                                                          neg_poss, 
                                                                                          word_tags, 
                                                                                          prec_punct_tags,
                                                                                          succ_punct_tags, 
                                                                                          merge_punct)

                if updated_tree is not None:
                    if len(new_children) == 0:
                        tree.start_idx = updated_tree.start_idx # set index correctly

                    new_children.append(updated_tree)

            if len(new_children) == 0:
                return None, prev_word_node, prev_left_punct
            
            else:
                tree.clear()
                tree.extend(new_children)
                return tree, prev_word_node, prev_left_punct
    
    @staticmethod
    def double_check_detokenize(sen : str, 
                                neg_poss : Sequence[str] = NEG_POSS, 
                                prec_punct_sym : Sequence[str] = INIT_PUNCT_SYM,
                                succ_punct_sym : Sequence[str] = TRAILING_PUNCT_SYM,
                                delimiter : str = ' ') -> str:
        """Manually corrects detokenization.

        Parameters
        ----------
        sen : str
            Detokenized sentence.
        neg_poss : Sequence[str], defaults to `NEG_POSS`
            Clitics that are split in the PTB
            tokenization.
        prec_punct_sym : Sequence[str], defaults to `INIT_PUNCT_SYM`
            Characters that should be treated as 
            preceeding punctuation.
        succ_punct_sym : Sequence[str], defaults to `TRAILING_PUNCT_SYM`
            Characters that should be treated as
            succeeding punctuation.
        delimiter : str, defaults to ' '
            Character(s) to split the sentence at.

        Returns
        -------
        str
            The corrected sentence.
        """
        # Note: Implementation using regex would probably eliminate 
        # the last 10 errors on the PTB train split

        words : List[str]
        # Merge clitics that are not split yet
        words = sen.split(delimiter)

        i : int

        # Re-split contractions that should not
        # be merged
        i = 0
        while i < len(words):
            if words[i].lower().startswith('cannot'):
                words.insert(i+1, words[i][3:])
                words[i] = words[i][:3]

            if words[i].lower().endswith('cannot'):
                words.insert(i+1, words[i][-3:])
                words[i] = words[i][:-3]

            elif words[i].lower().endswith('gonna'):
                words.insert(i+1, words[i][-2:])
                words[i] = words[i][:-2]

            elif words[i].lower().startswith('gonna'):
                words.insert(i+1, words[i][3:])
                words[i] = words[i][:-3]
            i += 1

        # Treat three dots
        i = 0
        while i < len(words):
            split = words[i].split("...")
            if len(split) == 2:
                # Concat to preceeding word if it is not punctuation and if the dots
                # are not followed by additional dots

                if (not split[1].startswith(".") 
                    and not split[0] in prec_punct_sym 
                    and not split[0] in succ_punct_sym):

                    words[i] = split[0] + "..."
                    words.insert(i+1, split[1])
            i += 1

        # Treat clitics that are still tokenized and punctuation
        i = 0
        while i < len(words):
            
            # suceeding clitics and punctuation
            if ((words[i] in neg_poss 
                    or words[i] in succ_punct_sym 
                    or all([c in succ_punct_sym for c in words[i]])) 
                and not i == 0):
                
                # append to preceeding token
                words[i-1] = words[i-1] + _replace_punct(words[i])
                del words[i]
            
            # preceeding punctuation or succeeding punctuation if at start of the sentence
            elif (words[i] in prec_punct_sym 
                  or all([c in prec_punct_sym for c in words[i]]) 
                  or (words[i] in succ_punct_sym and i == 0) 
                  or words[i] == '"`'): # manual correction
                
                # prepend
                words[i+1] = _replace_punct(words[i]) + words[i+1]
                del words[i]

            # treatment of brackets
            elif words[i][:3] == "RRB" or words[i][:3] == "RCB":
                words[i-1] = words[i-1] + _replace_punct(words[i][:3]) + words[i][3:]
                del words[i]

            elif words[i][-3:] == "LRB" or words[i][-3:] == "LCB":
                words[i+1] =  words[i][:-3] + _replace_punct(words[i][-3:]) + words[i+1]
                del words[i]

            else:
                i += 1
        
        sen = delimiter.join(words)
        words = sen.strip().split(delimiter)

        # Treat em-dash: We tokenize it since this corresponds to the
        # tree transformation
        i = 0
        while i < len(words):
            split = words[i].split("--")
            if len(split) >= 2 and not (len(split[0]) == 0 or len(split[1]) == 0):
                words[i] = split[0]
                words.insert(i+1, "--")
                words.insert(i+2, "--".join(split[1:]))
            i += 1

        # Correct mistake that the PTB string detokenizer caused:
        # Replace "' with '". 
        i = 0
        while i < len(words):
            if len(words[i]) == 1:
                pass
            else:
                word_new : str = ""
                last_char : str = ""
                for j in words[i]:
                    if last_char == "\"" and j == "'":
                        word_new += "'"
                        last_char = "\""
                    else:
                        word_new += last_char
                        last_char = j
                else:
                    word_new += last_char
                    
                words[i] = word_new
            i += 1
        
        sen = delimiter.join(words)

        return sen.strip()

    @property
    def words(self) -> List[str]:
        """The tree leaves which have
        a parent node with a POS tag as label.
        """
        return self.extract_words()
    
    def extract_words(self, lower : bool = False, 
                      word_tags : Sequence[str] = WORD_TAGS) -> List[str]:
        """Extracts the word leaves from 
        the tree. Omits punctuation and
        traces.

        Words are identified by their parent
        label, i.e. POS tag.

        Parameters
        ----------
        lower : bool, defaults to False
            Whether to convert the words to lowercase.
        word_tags : Sequence[str], defaults to `WORD_TAGS`
            Tags to be considered POS tags.

        Returns
        -------
        List[str]
            Extracted words.
        """
        words : List[str] = []
        word : str
        tag : str

        for word, tag in self.tree.pos():
            if tag in word_tags:
                if lower:
                    word = word.lower()

                words.append(word)
        
        return words
    
    def remove_non_words(self, inplace : bool = False,
                         word_tags : Sequence[str] = WORD_TAGS) -> Self:
        """Removes all nodes that have
        no word leaf. Works inplace.
        
        Parameters
        ----------
        inplace : bool, defaults to True
            Whether to perform some of the transformations
            in the existing objects.
        word_tags : Sequence[str], defaults to `WORD_TAGS`
            Tags to be considered POS tags.

        """

        def _remove_recursive(node : Tree) -> bool:
            """Recusively removes all nodes
            that dominate only non-word leaves.

            Parameters
            ----------
            node : Tree
                Tree to perform the action on.

            Returns
            -------
            bool
                True if the node dominates
                only non-word leaves or no
                leaves.
            """
            if len(node) == 1 and isinstance(node[0], str):
                # if there is only one string daughter
                # the node is a leaf

                if node.label() in word_tags:
                    return False
                    
                else:
                    return True
                
            else:
                for i, child in sorted(enumerate(node),         # pop all child nodes
                                       reverse = True,          # that don't dominate 
                                       key = lambda x : x[0]):  # words.
                    # The iteration is done in reverse order so that popping
                    # does not mix up the running index.

                    if _remove_recursive(child) is True:
                        node.pop(i)

                if len(node) == 0:
                    return True     # If the node has no daughters, it can also be left out.
                
                else:
                    return False

        tree_to_alter : Tree = self.tree if inplace else self.tree.deepcopy()

        _remove_recursive(tree_to_alter)
            
        return self.__class__(string = None, tree = tree_to_alter, detokenized = self.detokenized)

    def remove_punctuation(self, inplace : bool = False, use_tags : bool = False,
                           punct_tags : Sequence[str] = PUNCT_TAGS) -> Self:
        """Creates a new Representation without punctuation.
        
        Parameters
        ----------
        inplace : bool, defaults to False
            Whether to perform some of the
            tree altering operations inplace to
            speed up the process.
        use_tags : bool, defaults to False
            Whether to use POS tags to determine
            punctuation (if True) or use the naive algorithm
            of the `TreeRepresentation` parent class (if False).
        punct_tags : Sequence[str], defaults to PUNCT_TAGS
            POS tags that indicate the leaves
            that should be removed.
        
        Returns
        ----------
        Self
            The representation.
        """

        def _remove_punct_leaves(tree : Tree) -> bool:
            """Recursively remove punctuation leaves from
            tree. Performs inplace-action.
            
            Parameters
            ----------
            tree : Tree
                Tree node.

            Returns
            ----------
            bool
                Whether to remove this node. 
                
            """
            if tree.label() in punct_tags:
                return True

            else:
                for i, child in reversed(list(enumerate(tree))):
                    if isinstance(child, str):
                        continue

                    if _remove_punct_leaves(child):
                        tree.pop(i)
                
                if len(tree) == 0:
                    return True
                
                else:
                    return False

        if use_tags:
            tree_to_alter : Tree = self.tree
            if not inplace:
                tree_to_alter = tree_to_alter.deepcopy()

            _remove_punct_leaves(tree_to_alter)

            return self.__class__(string = None,
                                  tree = tree_to_alter,
                                  detokenized = self.detokenized)

        else:
            new_representation : Self = super().remove_punctuation(inplace = inplace)
            new_representation.detokenized = self.detokenized
            return new_representation
    
    def remove_traces(self, inplace : bool = False) -> Self:
        """Remove trace nodes. All words that begin with *
        are treated as traces.
        
        Parameters
        ----------
        inplace : bool, defaults to False
            Whether to perform some of the tree
            transformations inplace. Still creates
            a new representation object.

        Returns
        -------
        Self
            The new represenntation
        """

        if inplace:
            self.tree.remove_traces(inplace = True)
            return self.__class__(string = None, tree = self.tree,
                                  detokenized = self.detokenized)
        
        else:
            tree_copy : Tree = self.tree.deepcopy()
            tree_copy = tree_copy.remove_traces(inplace = False)
            return self.__class__(string = None, tree = tree_copy,
                                  detokenized = self.detokenized)


def write_trees(trees : Iterable[Tree], file_path : str,
                brackets : str = "[]") -> None:
    """Writes a sequence of
    `nltk.Tree` objects to a file.
    The trees are separated by line breaks.

    Parameters
    ----------
    trees : Iterable[NltkTree]
        The sequence of trees to
        write to the file.
    file_path : str
        File path to use
    """
    first_written : bool = False
    # Keep track of the first time a tree was written
    # to prevent a line break to be written at the start
    # of the file.

    with open(file_path, "w") as file:
        for tree in trees:
            if first_written:
                file.write("\n")

            file.write(tree.str_representation(brackets))

            first_written = True

def _replace_punct(w : str) -> str:
    """Utility method to replace
    PTB tokenized characters with
    their true forms.
    
    Parameters
    ----------
    w : str
        Token.

    Returns
    ----------
    str
        The true form.
    """
    match_dict : Dict[str, str] = {'``' : '"',
                                   "''" : '"',
                                   "RCB" : ")",
                                   "RRB" : ")",
                                   "LCB" : "(",
                                   "LRB" : "("}
    
    if w in match_dict.keys():
        return match_dict[w]
    
    else:
        return w
        
class TreeTransformError(AssertionError):
    def __init__(self, message):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)