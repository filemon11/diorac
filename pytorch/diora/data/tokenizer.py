"""Features 

Classes
----------
TreeTokenizer
    Tokenizes (tree) representations and produces
    indices (and transformed trees).
"""

from diora.data.segmenter import TreeSegmenter, TokenizerMode
from diora.data.segmenter import TreeRepresentation, Representation, TokenizedRepresentation
from diora.data.segmenter import FutureWord2i as Word2i

from diora.data.treeencoding import TreeEncoding, FlatEncoding, BatchEncoding

from typing import Union, overload, List, Iterable, cast, Iterable, Optional, Literal, Sequence
from typing_extensions import Self


class TreeTokenizer():
    """Tokenizes (tree) representations and produces
    indices (and transformed trees).

    Attributes
    ----------
    segmenter : TreeSegmenter
        A segmenter that splits representations
        into subsequences.
    word2i : Word2i
        A mapping that maps subsequences to
        indices.
    """
    def __init__(self, segmenter : TreeSegmenter, word2i : Word2i):
        """Initialises a `TreeTokenizer`.
        
        Arguments
        ----------
        segmenter : TreeSegmenter
            A segmenter that splits representations
            into subsequences.
        word2i : Word2i
            A mapping that maps subsequences to
            indices.
        """

        self.segmenter : TreeSegmenter = segmenter
        """A segmenter that splits representations
        into subsequences."""

        self.word2i : Word2i = word2i
        """A mapping that maps subsequences to
        indices.
        """

    def tokenize(self, sentence : Union[TreeRepresentation,
                                        Representation]) -> List[str]:
        """Split a sentence into a list of subsequences.

        Parameters
        ----------
        sentence : Union[TreeRepresentation, Representation]
            The input sequence.

        Returns
        -------
        List[str]
            The subsequences generated.
        """
        
        tokenized : TokenizedRepresentation = self.segmenter(sentence)

        return tokenized.tokens
    
    def encode(self, sentence : Union[TreeRepresentation,
                                        Representation]) -> List[int]:
        """Convert a sentence into a sequence
        of indices by tokenizing it and mapping
        each token to an index.

        Parameters
        ----------
        sentence : Union[TreeRepresentation, Representation]
            The input sequence.

        Returns
        -------
        List[int]
            The index sequence generated.
        """
        return self.word2i[self.tokenize(sentence)]
    
    @overload
    @classmethod
    def from_pretrained(cls, tokenizer_name : str, 
                        w2i_from_tokenizer : Literal[True] = True, 
                        init_add : bool = True,
                        load_word2i_dir : None = None) -> Self:
        ...

    @overload
    @classmethod
    def from_pretrained(cls, tokenizer_name : str, 
                        w2i_from_tokenizer : Literal[False], 
                        init_add : bool = True, 
                        load_word2i_dir : Optional[str] = None) -> Self:
        ...

    @classmethod
    def from_pretrained(cls, tokenizer_name : str, 
                        w2i_from_tokenizer : bool = True, 
                        init_add : bool = True,
                        load_word2i_dir : Optional[str] = None) -> Self:
        """Load a `TreeTokenizer` based on a pretrained
        huggingface tokenizer and optionally a saved
        `Word2i` mapping.

        Parameters
        ----------
        name : str
            Huggingface path of the tokenizer
            to load.
        w2i_from_tokenizer : bool, defaults to True
            Whether to load vocabulary from 
            the tokenizer.
        init_add : bool, defaults to True
            Whether to initialise the `Word2i`
            mapping directly with the complete
            tokenizer vocabulary. Irrelevant
            if `w2i_from_tokenizer = False`.
        load_word2i_dir : Optional[str], defaults to None
            Path to the word2i directory
            to load. Must be None if `w2i_from_tokenizer`
            is set to True. If not `w2i_from_tokenizer`
            and `load_word2i_dir` is None then 
            a new `Word2i` mapping is created.

        Returns
        -------
        Self
            A new `TreeTokenizer`.
        """
        tokenizer : TreeSegmenter = TreeSegmenter.from_pretrained(tokenizer_name)

        word2i : Word2i
        
        if w2i_from_tokenizer:
            word2i = Word2i(tokenizer, init_add = init_add)

        elif load_word2i_dir is None:
            word2i = Word2i()
        
        else:
            word2i = Word2i.load(load_word2i_dir)
        
        return cls(tokenizer, word2i)
    
    @overload
    def __call__(self, sentence : Union[Iterable[TreeRepresentation], TreeRepresentation],
                 add_to_dict : bool = False,
                 mode : TokenizerMode = "simple",
                 add_future : bool = False,
                 inplace : bool = False) -> TreeEncoding:
        ...

    @overload
    def __call__(self, sentence : Union[Iterable[Representation], Representation],  # type: ignore
                 add_to_dict : bool = False,
                 mode : TokenizerMode = "simple",
                 add_future : bool = False,
                 inplace : bool = False) -> FlatEncoding:                           # Does it match because str > TreeRepresentation
        ...                                                                         # and str = Iterable[str] > Iterable[Representation]?

    def __call__(self, sentence : Union[TreeRepresentation,
                                        Representation,
                                        Iterable[TreeRepresentation],
                                        Iterable[Representation]],
                 add_to_dict : bool = False,
                 mode : TokenizerMode = "simple",
                 add_future : bool = False,
                 inplace : bool = False,
                 filter_length : Optional[None] = None,
                 replace_rare : Optional[int] = None) -> Union[TreeEncoding, FlatEncoding]:
        """Convert a representation or an iterable of
        representations into an encoding that
        contains the indices, attention mask and 
        optionally transformed trees, as a product
        of tokenization.

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
        sentence : Union[TreeRepresentation, Representation, 
                         Iterable[TreeRepresentation], Iterable[Representation]]
            The sequence(s) to tokenize.
        add_to_dict : bool, optional
            Whether to add unseen tokens to the
            word2i dictionary.
        mode : Literal['simple', 'word', 'sentence', 'fragment'],
               defaults to 'simple'.
            Controls tokenization mode.
        add_future : bool, defaults to False
            Whether to add special future tag.
        inplace : bool, defaults to False
            Whether to perform some tree 
            transformations inplace.
        filter_length : Optional[int], defaults to None
            If specified and a list of sentences is given,
            sentences of greater length are missing from
            the output list.
        replace_rare : Optional[int], defaults to None
            If add_to_dict is True, specifies 
            a certain absolute occurrence up to which
            a token should be replaced by the unkown
            token.

        Returns
        -------
        Union[TreeEncoding, FlatEncoding]
            The resulting encoding. A dictionary
            that contains the tokenization results.
        """

        tokenized : Union[TokenizedRepresentation, List[TokenizedRepresentation],
                          TreeRepresentation, List[TreeRepresentation]]
        encoded : Union[List[int], List[List[int]]]
        attention_mask : Union[List[int], List[List[int]]]

        if (not isinstance(sentence, str)
                and isinstance(sentence, Iterable) 
                and isinstance(next(iter(sentence)), Representation)):
            # If several representations are given.

            tokenized = [self.segmenter(rep, 
                                        mode = mode,
                                        add_future = add_future,
                                        inplace = inplace) for rep in sentence]
            
            if filter_length is not None:
                # Filter out sentences over certain length.
                tokenized = [sentence for sentence in tokenized if len(sentence.tokens) <= filter_length]

            if replace_rare is not None and add_to_dict:
                # Rare replacement.
                # Prepare Word2i mapping first; then delete all rare tokens.

                self.word2i.get_idx([t.tokens for t in tokenized], add_to_dict = True)

                self.word2i.del_rare(replace_rare)
                
                encoded = self.word2i.get_idx([t.tokens for t in tokenized], add_to_dict = False)

            else:
                encoded = self.word2i.get_idx([t.tokens for t in tokenized], 
                                              add_to_dict = add_to_dict)

            encoded = cast(List[List[int]], encoded)    # weird error...
            attention_mask = [[1] * len(l) for l in encoded]
        
        else:
            # If a single representation is given.
            tokenized = self.segmenter(sentence, 
                                       mode = mode,
                                       add_future = add_future,
                                       inplace = inplace)
            
            if replace_rare is not None and add_to_dict:
                # Rare replacement.
                # Prepare Word2i mapping first; then delete all rare tokens.

                self.word2i.get_idx(tokenized.tokens, add_to_dict = True)

                self.word2i.del_rare(replace_rare)
                
                encoded = self.word2i.get_idx(tokenized.tokens, add_to_dict = False)

            else:
                encoded = self.word2i.get_idx(tokenized.tokens, 
                                          add_to_dict = add_to_dict)
            
            attention_mask = [1] * len(encoded)

        if isinstance(tokenized, TreeRepresentation):
            # If it is a single tree.
            return BatchEncoding({"input_ids" : encoded,
                                  "attention_mask" : attention_mask,
                                  "tokens" : tokenized.tokens,
                                  "trees" : tokenized.tree})
        
        elif (isinstance(tokenized, list)
                and len(tokenized) > 0 
                and isinstance(tokenized[0], TreeRepresentation)):
            # If it is a sequence of trees.

            return BatchEncoding({"input_ids" : encoded,
                                  "attention_mask" : attention_mask,
                                  "tokens" : [t.tokens for t in tokenized],
                                  "trees" : [t.tree for t in tokenized]})
        
        elif isinstance(tokenized, Representation):
            return BatchEncoding({"input_ids" : encoded,
                                  "tokens" : tokenized.tokens,
                                  "attention_mask" : attention_mask})
        
        else:
            return BatchEncoding({"input_ids" : encoded,
                                  "tokens" : [t.tokens for t in tokenized],
                                  "attention_mask" : attention_mask})


