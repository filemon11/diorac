"""Module for composing datasets for unlabelled
parsers both for training and prediction.

Methods
----------

"""

from diora.data.representations import Representation, TreeRepresentation, TokenizedRepresentation, Tree
from diora.data.readers import construct_reader, FileType, PTBReaderOptions, ListTreeReaderOptions, FlatReaderOptions, Reader
from diora.data.segmenter import Word2i, Char2i
from diora.data.tokenizer import TreeTokenizer
from diora.data.batch_iterator import make_batch_iterator, BatchIterator, IteratorOptions, LossOptions
from diora.data.treeencoding import CharEncoding, CharTreeEncoding, FlatEncoding, TreeEncoding

import torch
from random import random

from typing import List, Tuple, Optional, TypeVar, Literal, Union, overload, Dict


RP = TypeVar("RP", bound = Representation)
TRP = TypeVar("TRP", bound = TokenizedRepresentation)

def extend_future(tokenized : Union[TreeEncoding, FlatEncoding], 
                  add_fragments : Literal["all", "random"],
                  min_split_idx : int = 3) -> None:
    """Produce fragments from encoded sentences.

    Options:
    - 'all': For every sentence in the input, create
    a fragment for every sentence position.
    - 'random': For every sentence in the input,
    create only one fragment at a random position.

    Parameters
    ----------
    tokenized : Union[TreeEncoding, FlatEncoding]
        The encoded corpus.
    add_fragments : Literal['all', 'random']
        Whether to take all possible positions
        for fragmentation or one random split
        per sentence.
    min_split_idx : int, defaults to 3
        Smallest index allowed for splitting.
        Value of 3 means that sentences with
        trivial trees of 1 or 2 leaves are 
        excluded.

    Returns
    -------
    None
    """

    @overload
    def _limit_to(sequence : List, index : int, retain_last : bool) -> List:
        ...

    @overload
    def _limit_to(sequence : TokenizedRepresentation, index : int, retain_last : bool) -> TokenizedRepresentation:
        ...

    def _limit_to(sequence : List | TokenizedRepresentation, index : int, retain_last : bool) -> List | TokenizedRepresentation:
        """Cut a list or a `TokenizedRepresentation`
        off at a certain position. Creates a (shallow)
        copy of the sequence.
        
        Parameters
        ----------
        sequence : List | TokenizedRepresentation
            The sequence to cut off.
        index : int
            The cut-off point. Everything right
            of it (including the point itself)
            is discarded in the output.
        retain_last : bool
            Whether to retain the last
            list or representation element
            (here: future token).

        Returns
        -------
        List | TokenizedRepresentation
            The new sequence, not including
            the cut-off part.
        """
        if isinstance(sequence, TokenizedRepresentation):
            return sequence.limit_to(index, retain_last = retain_last)
        else:
            if retain_last:
                return sequence[:index] + sequence[-1:]
            else:
                return sequence[:index]

    fragments_positions : List[List[int]] = []
    if add_fragments == "all":
        for sentence in tokenized["tokens"]:
            fragments_positions.append([i+1 for i in range(min_split_idx, len(sentence)-1)])

    elif add_fragments == "random":
        for sentence in tokenized["tokens"]:
            position = int(random() * (len(sentence) - min_split_idx)) +1
            fragments_positions.append([position + min_split_idx])
        
    else:
        raise Exception("Unknown add_fragments value.")
    
    
    for category, sentence_list in tokenized.items():
        new_sentences = []
        for j, (sen_fragment_positions, sen) in enumerate(zip(fragments_positions, sentence_list)):
            new_sentences.extend([_limit_to(sen, position, True) for position in sen_fragment_positions])
        sentence_list.extend(new_sentences)

def filter_encoding(encoding : Union[TreeEncoding, FlatEncoding,
                                     CharEncoding, CharTreeEncoding],
                    filter_length : int) -> None:
    """Deprecated"""
    
    include_map : List[bool] = [(0 < len(sentence) <= filter_length)
                                        for sentence in encoding["input_ids"]]
    
    for key in encoding.keys():
        encoding[key] = [entry for entry, include in zip(encoding[key], include_map) if include]
    
@overload
def create_dataset(filetype : Literal["flat"],
                   filename : str,
                   tokenizer : TreeTokenizer,
                   char2i : Char2i,
                   filter_length : Optional[int] = None,
                   add_to_dict : bool = False,
                   detokenize : bool = True,
                   remove_punct : bool = True,
                   add_fragments : Literal["all", "random", False] = False,
                   min_length : int = 3) -> CharEncoding:
    ...

@overload
def create_dataset(filetype : Literal["ptb"],
                   filename : str,
                   tokenizer : TreeTokenizer,
                   char2i : Char2i,
                   filter_length : Optional[int] = None,
                   add_to_dict : bool = False,
                   detokenize : bool = True,
                   remove_punct : bool = True,
                   add_fragments : Literal["all", "random", False] = False,
                   min_length : int = 3) -> CharTreeEncoding:
    ...

@overload
def create_dataset(filetype : Literal["flat"],  # type: ignore
                   filename : str,
                   tokenizer : TreeTokenizer,
                   char2i : None = None,
                   filter_length : Optional[int] = None,
                   add_to_dict : bool = False,
                   detokenize : bool = True,
                   remove_punct : bool = True,
                   add_fragments : Literal["all", "random", False] = False,
                   min_length : int = 3) -> FlatEncoding:
    ...

@overload
def create_dataset(filetype : Literal["ptb"],   # type: ignore
                   filename : str,
                   tokenizer : TreeTokenizer,
                   char2i : None = None,
                   filter_length : Optional[int] = None,
                   add_to_dict : bool = False,
                   detokenize : bool = True,
                   remove_punct : bool = True,
                   add_fragments : Literal["all", "random", False] = False,
                   min_length : int = 3) -> TreeEncoding:
    ...

def create_dataset(filetype : FileType,
                   filename : str,
                   tokenizer : TreeTokenizer,
                   char2i : Optional[Char2i] = None,
                   filter_length : Optional[int] = None,
                   add_to_dict : bool = False,
                   detokenize : bool = True,
                   remove_punct : bool = True,
                   add_fragments : Literal["all", "random", False] = False,
                   min_length : int = 3,
                   add_future : bool = False) \
                    -> Union[TreeEncoding, FlatEncoding,
                             CharEncoding, CharTreeEncoding]:
    
    options : PTBReaderOptions | ListTreeReaderOptions | FlatReaderOptions 
    match filetype:
        case "flat":
            options = FlatReaderOptions()

        case "ptb":
            options = PTBReaderOptions(detokenize = detokenize,
                                       remove_non_words = remove_punct,
                                       remove_traces = True)
        
        case "json":
            options = ListTreeReaderOptions(key = "tree",
                                            brackets = "()")
        
        case _:
            raise Exception(f"Filetype {str(filetype)} not know.")
        
    
    reader : Reader[Representation] = construct_reader(filetype, 
                                                       filename, 
                                                       options,
                                                       min_length = min_length)
    
    sentences : List[Representation] = list(reader)
    print(f"Sentence list loaded. Contains {len(sentences)} sentences.")
    
    if remove_punct:
        sentences = [r.remove_punctuation() for r in reader]
        print(f"Punctuation removed.")

    inplace : bool = isinstance(sentences[0], TreeRepresentation)

    # General tokenization
    tokenized : Union[TreeEncoding, FlatEncoding] = tokenizer(sentences, 
                                                              add_to_dict, 
                                                              inplace = inplace,
                                                              mode = "simple",  # set to "sentence" for LLM mode
                                                              add_future = add_future,
                                                              filter_length = filter_length,
                                                              replace_rare = 1)

    
    print(f"Tokenization done for {len(tokenized['input_ids'])} sentences.")
    
    if add_fragments is not False:
        extend_future(tokenized, add_fragments)
        print(f"Future extension done.")

    
    # Character indices
    if char2i is not None:
        char_idxs = char2i.chars.get_idx(tokenized["tokens"], add_to_dict = add_to_dict)
        tokenized["char_ids"] = char_idxs
        print("Chars created.")

    ## Filter
    #if filter_length is not None:
    #    filter_encoding(tokenized, filter_length)

    #print([s[2] for s in tokenized["input_ids"]])
    return tokenized


def get_iterator(dataset : FlatEncoding | CharEncoding, word2i : Word2i, char2i,
                 filter_length, iterator_options : IteratorOptions, loss_options : LossOptions,
                 mode : Literal["train", "validation"]) -> BatchIterator:

    shuffle : bool
    include_partial : bool

    match mode:
        case "train":
            shuffle = True
            include_partial = False

        case "validation":
            shuffle = True                  # Before: shuffle False, include_partial True
            include_partial = False         # But this setting necessary for hyperopt

        case _:
            raise Exception(f"Mode {str(mode)} unknown. Only 'train' or 'validation' allowed")
        
    return make_batch_iterator(dataset, word2i, char2i, shuffle=shuffle, include_partial = include_partial,
                               iterator_options = iterator_options, loss_options = loss_options,
                               filter_length = filter_length)


def prepare_train_val(train_file_type : FileType,
                      train_file : str,
                      val_file_type : FileType,
                      val_file : str,
                      tokenizer_name : str,
                      word2i_dir : str,
                      char2i_dir : str,
                      iterator_options : IteratorOptions,
                      loss_options : LossOptions,
                      train_filter_length : int,
                      validation_filter_length : int,
                      load_w2i_from_tokenizer : bool = True,
                      take_all_from_w2i : bool = True,
                      embeddings_type : Literal["word", "hybrid"] = "word",
                      fragments : Literal[False, "random", "all"] = False,
                      add_future : bool = False,
                      use_validation : bool = True
                      ) -> Tuple[BatchIterator,
                                 BatchIterator | None,
                                 Word2i,
                                 Optional[Char2i]]:

    tokenizer : TreeTokenizer

    tokenizer = TreeTokenizer.from_pretrained(tokenizer_name,
                                              load_w2i_from_tokenizer,
                                              take_all_from_w2i)
    
    char2i : Optional[Char2i] = None
    if embeddings_type == "hybrid":
        char2i = Char2i()

    train_encoding : CharEncoding | FlatEncoding 
    train_encoding = create_dataset(train_file_type,
                                    train_file,
                                    tokenizer,
                                    char2i,
                                    train_filter_length,
                                    add_to_dict = True,
                                    detokenize = True,
                                    remove_punct = False,
                                    add_fragments = fragments,
                                    min_length = 3,
                                    add_future = add_future)
    
    print(train_encoding["tokens"][0])
    
    print("Num of sentences:", len(train_encoding["input_ids"]))

    tokenizer.word2i.save(word2i_dir)

    if embeddings_type == "hybrid":
        assert char2i is not None
        char2i.save(char2i_dir)
    
    train_iterator : BatchIterator = get_iterator(
        train_encoding,
        tokenizer.word2i, 
        char2i,
        train_filter_length,
        iterator_options,
        loss_options,
        mode = "train" #TODO
    )

    val_iterator : BatchIterator | None = None
    if use_validation:
        val_encoding : CharEncoding | FlatEncoding
        val_encoding = create_dataset(val_file_type,
                                      val_file,
                                      tokenizer,
                                      char2i,
                                      validation_filter_length,
                                      add_to_dict = False,
                                      detokenize = True,
                                      remove_punct = False,
                                      add_fragments = False,
                                      min_length = 0,
                                      add_future = add_future)
        
        val_iterator = get_iterator(
            val_encoding,
            tokenizer.word2i,
            char2i,
            validation_filter_length,
            iterator_options,
            loss_options,
            mode = "validation"
        )

    return train_iterator, val_iterator, tokenizer.word2i, char2i

def prepare_test(test_file_type : FileType,
                 test_file : str,
                 tokenizer_name : str,
                 word2i_dir : str,
                 char2i_dir : str,
                 iterator_options : IteratorOptions,
                 loss_options : LossOptions,
                 validation_filter_length : Optional[int] = None,
                 ) -> Tuple[List[Tree],
                            BatchIterator,
                            Word2i,
                            Optional[Char2i]]:
    
    
    tokenizer = TreeTokenizer.from_pretrained(tokenizer_name,
                                              w2i_from_tokenizer = False,
                                              init_add = False,
                                              load_word2i_dir = word2i_dir)
    char2i : Char2i | None
    try:
        char2i = Char2i.load(char2i_dir)

    except FileNotFoundError:
        char2i = None

    test_encoding = create_dataset(test_file_type,
                                   test_file,
                                   tokenizer,
                                   char2i,
                                   filter_length = validation_filter_length,
                                   add_to_dict = False,
                                   detokenize = True,
                                   remove_punct = False,
                                   add_fragments = False,
                                   min_length = 0)
    
    # deal with trees
    test_trees = test_encoding["trees"] if "trees" in test_encoding.keys() else None

    test_iterator : BatchIterator = get_iterator(
        test_encoding,
        tokenizer.word2i, 
        char2i,
        validation_filter_length,
        iterator_options,
        loss_options,
        mode = "validation"
    )

    return test_trees, test_iterator, tokenizer.word2i, char2i
    
def make_tensor(sentences : List[List[int]]) -> List[torch.Tensor]:
    return [torch.tensor(sentence, dtype=torch.int) for sentence in sentences]
