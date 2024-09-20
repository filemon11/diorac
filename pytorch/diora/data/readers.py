"""A collection of classes that
allow reading and parsing a variety of
file formats that contain syntactic information.

Classes
----------
PreReader
    Abstract class that prescribes `PreReader` subclass methods.
PreLineReader
    Reads one line at a time from a text file.
PreJSONReader
    Reads one string at a time from a JSONL file.
Reader
    Abstract class defining methods to be implemented by readers.
FlatReader
    Reader for simple text files with one sentence per line.
PTBReader
    Reader for PennTreebank (PTB) trees with one bracketed tree per line.

Methods
----------
construct_reader(filetype, filename, detokenize)
    Construct a `Reader` for a given file type.
"""


from diora.data.representations import Representation, PTBTreeRepresentation, TreeTransformError, TreeRepresentation
from diora.data.exceptions import check_bracket_format

import json

from typing import Generic, Literal, Iterator, Union, Optional, List, TypedDict, overload
from typing import TypeVar, cast
from typing_extensions import NotRequired

from abc import ABC, abstractmethod

FileType = Literal["ptb", "flat", "json"]
"""Possible file types for readers.
- 'ptb': PennTreebank (PTB), one tree per line
- 'flat': simple string, one sequence per line
- 'json': JSONL file with some attribute that contains 
`ListTree`s"""

ListTree = List[Union[str, "ListTree"]]
"""Diora format for unlabelled trees."""

RP = TypeVar("RP", bound = "Representation")
T = TypeVar("T")

GENERIC_NODE = "S"
"""Label to use for internal nodes
when converting from unlabelled `ListTree` format to
`TreeRepresentation`"""

GENERIC_LEAF = "DT"
"""abel to use for pre-terminal nodes
when converting from unlabelled `ListTree` format to
`TreeRepresentation`"""


class PreReader(ABC, Generic[T]):
    """Abstract class that defines prerequisites
    for `PreReader` subclasses. Pre-readers should
    be iterable objects that retrieve some kind of
    representation usable by a `Reader` object from
    a pre-specified location.

    Attributes
    ----------
    filename : str
        The path to the file that should be read.
    """

    def __init__(self, filename : str):
        """Initialise `PreReader`.

        Parameters
        ----------
        filename : str
            A path to the file that should be read.
        """
        self.filename : str = filename
        """The path to the file to be read."""

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        ...

class PreLineReader(PreReader[str]):
    """Pre-reader that reads one line
    at a time from a file.

    Attributes
    ----------
    strip_whitespace : str
        Whether to strip whitespace from
        each line.
    """
    def __init__(self, filename : str, strip_whitespace : bool = True):
        """Initialises PreLineReader.

        Parameters
        ----------
        filename : str
            The path to the file to be read.
        strip_whitespace : bool, defaults to True
            Whether to strip whitespace from
            each line.
        """
        super().__init__(filename)

        self.strip_whitespace : bool = strip_whitespace
        """Whether to strip whitespace from
        each line."""
    
    def __iter__(self) -> Iterator[str]:
        """Reads one line from a file
        at a time.

        Yields
        ------
        Iterator[str]
            The line iterator.
        """
        with open(self.filename, "r") as file:

            for line in file:

                if self.strip_whitespace:
                    yield line.strip()

                else:
                    yield line


class PreJSONReader(PreReader[ListTree]):
    """A pre-reader that reads JSONL files
    that contain `ListTree`s.

    Attributes
    ----------
    key : str
        The field in each line that contains
        the `ListTree`.
    """
    def __init__(self, filename : str, key : str):
        """Initialises a `PreJSONReader`.

        Parameters
        ----------
        filename : str
            The path to the file to read.
        key : str
            The field that contains the `ListTree`.
        """
        super().__init__(filename)
        self.key : str = key
        """The field that contains the `ListTree`."""

    def __iter__(self) -> Iterator[ListTree]:
        """Iterate through the JSONL file
        returning one `ListTree` at a time.

        Yields
        ------
        Iterator[ListTree]
            _description_
        """
        with open(self.filename, "r") as file:
            for line in file:
                yield json.loads(line)[self.key]

class Reader(Generic[RP], ABC):
    """Lets you iterate over file giving one
    representation at a time.
    """
    @abstractmethod
    def __init__(self, filename : str) -> None:
        """Should receive a filename.

        Parameters
        ----------
        filename : str
            Path to file.
        """
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[RP]:
        """Should be iterable
        and return a `Representation
        at each step.`

        Yields
        ------
        Iterator[RP]
            A representation taken from one
            line of the file.
        """
        ...

class LineReader(Reader[RP], ABC):
    """Abstract specifications for a reader
    that can build representations from 
    simple line strings.

    Attributes
    ----------
    pre_line_reader : PreLineReader
        The pre-line reader to use
        to retrieve information from
        the file.
    """

    @overload
    def __init__(self, filename : str,
                 strip_whitespace : bool,
                 pre_line_reader : None = None):
        ...

    @overload
    def __init__(self, filename : None,
                 strip_whitespace : None,
                 pre_line_reader : PreLineReader):
        ...

    def __init__(self, filename : Optional[str] = None, 
                 strip_whitespace : Optional[bool] = None,
                 pre_line_reader : Optional[PreLineReader] = None):
        """Initialise `LineReader`. You can either
        specify a `pre_line_reader` or let the
        initialiser build a simple one (`PreLineReader`)
        by specifying `filename` and `strip_whitespace`.

        Parameters
        ----------
        filename : Optional[str], defaults to None
            The path to the file to read.
        strip_whitespace : Optional[bool], defaults to None
            Whether to strip whitespace for each line.
        pre_line_reader : Optional[PreLineReader], defaults to None
            A custom pre-line reader.
        """
        
        self.pre_line_reader : PreLineReader
        """The pre-line reader to use for retrieving one string
        at a time from the file."""

        if pre_line_reader is None:
            # initialise a new PreLineReader
            assert isinstance(filename, str)
            assert isinstance(strip_whitespace, bool)

            self.pre_line_reader = PreLineReader(filename, strip_whitespace)

        else:
            # use the PreLineReader given as parameter
            assert filename is None and strip_whitespace is None

            self.pre_line_reader = pre_line_reader

class FlatReader(LineReader[Representation]):
    """A simple reader that reads
    one line at a time from a file
    and packs it in a flat string 
    representation without further
    modifications.

    Attributes
    ----------
    min_length : int
        The minimum length of words for retrieval.
        A word is defined as a sequence separated by
        the `delimiter` attribute value.
    delimiter : str
        Character(s) at which a split should be
        made to determine the number of elements.
    """
    @overload
    def __init__(self, filename : str,
                 strip_whitespace : bool,
                 pre_line_reader : None = None,
                 min_length : int = 0,
                 delimiter : str = ' '):
        ...

    @overload
    def __init__(self, filename : None,
                 strip_whitespace : None,
                 pre_line_reader : PreLineReader,
                 min_length : int = 0,
                 delimiter : str = ' '):
        ...

    def __init__(self, filename : Optional[str] = None, 
                 strip_whitespace : Optional[bool] = None,
                 pre_line_reader : Optional[PreLineReader] = None,
                 min_length : int = 0,
                 delimiter : str = ' '):
        """Initialise `LineReader`. You can either
        specify a `pre_line_reader` or let the
        initialiser build a simple one (`PreLineReader`)
        by specifying `filename` and `strip_whitespace`.

        Parameters
        ----------
        filename : Optional[str], defaults to None
            The path to the file to read.
        strip_whitespace : Optional[bool], defaults to None
            Whether to strip whitespace for each line.
        pre_line_reader : Optional[PreLineReader], defaults to None
            A custom pre-line reader.
        min_length : int, defaults to 0
            If > 0, splits each line at `delimiter`
            and retrieves only those representations
            that have at least `min_length` elements.
        delimiter : str, defaults to ' '
            Character(s) at which a split should be
            made to determine the number of elements.
        """
        super().__init__(filename, strip_whitespace, pre_line_reader) # type: ignore

        self.min_length : int = min_length
        """The minimum length of words for retrieval.
        A word is defined as a sequence separated by
        the `delimiter` attribute value."""

        self.delimiter : str = delimiter
        """Character(s) at which a split should be
        made to determine the number of elements."""


    def __iter__(self) -> Iterator[Representation]:
        """Reads one line from a flat
        text file at each step.

        Yields
        ------
        Iterator[Representation]
            The line.
        """

        for line in self.pre_line_reader:
            if len(line) > 0:
                # Do not produce empty representations

                if self.min_length == 0:
                    yield Representation(line)

                elif len(line.split(self.delimiter)) >= self.min_length:
                    yield Representation(line)
                

class PTBReader(LineReader[PTBTreeRepresentation]):
    """Reader for PennTreebank (PTB)
    files.

    Attributes
    ----------
    detokenize : bool
        Whether to detokenize the
        tree.
    remove_non_words : bool
        Whether to remove all non-word leaves
        and non-word-only subtrees.
    remove_traces : bool
        Whether to remove all trace leaves
        and trace-only subtrees.
    brackets : str
        Bracket format of the source
        file.
    min_length : int
        Minimum leaf count that a tree should
        have in order to be retrieved.
    """
    @overload
    def __init__(self, filename : str,
                 strip_whitespace : bool,
                 pre_line_reader : None,
                 detokenize : bool,
                 remove_non_words : Literal[True],
                 remove_traces : Literal[True] = True,
                 brackets : str = "()",
                 min_length : int = 0):
        ...

    @overload
    def __init__(self, filename : None,
                 strip_whitespace : None,
                 pre_line_reader : PreLineReader,
                 detokenize : bool,
                 remove_non_words : Literal[True],
                 remove_traces : Literal[True] = True,
                 brackets : str = "()",
                 min_length : int = 0):
        ...

    @overload
    def __init__(self, filename : str,
                 strip_whitespace : bool,
                 pre_line_reader : None = None,
                 detokenize : bool = True,
                 remove_non_words : Literal[False] = False,
                 remove_traces : bool = True,
                 brackets : str = "()",
                 min_length : int = 0):
        ...

    @overload
    def __init__(self, filename : None,
                 strip_whitespace : None,
                 pre_line_reader : PreLineReader,
                 detokenize : bool = True,
                 remove_non_words : Literal[False] = False,
                 remove_traces : bool = True,
                 brackets : str = "()",
                 min_length : int = 0):
        ...
        
    def __init__(self, filename : Optional[str] = None, 
                 strip_whitespace : Optional[bool] = None,
                 pre_line_reader : Optional[PreLineReader] = None,
                 detokenize : bool = True,
                 remove_non_words : bool = False,
                 remove_traces : bool = True,
                 brackets : str = "()",
                 min_length : int = 0) -> None:
        """Initialises a PTB reader. The file should contain one
        bracketed tree per line. You can either
        specify a `pre_line_reader` or let the
        initialiser build a simple one (`PreLineReader`)
        by specifying `filename` and `strip_whitespace`.

        Parameters
        ----------
        filename : str, defaults to None
            The path to the file that
            should be read.
        strip_whitespace : bool, defaults to None
            Whether to strip whitespace at each line.
        pre_line_reader : PreLineReader, defaults to None
            The pre-line reader to use
            to retrieve information from
            the file. 
        detokenize : bool, defaults to True
            Whether to detokenize the
            trees (i.e. remerge clitics that
            were split in the PTB corpus like
            ''You 're''.)
        remove_non_words : bool, defaults to False
            Whether to remove all non-word leaves
            and non-word-only subtrees.
        remove_traces : bool = True,
            Whether to remove all trace leaves
            and trace-only subtrees.
        brackets : str, defaults to '()'
            Brackets used in the source file.
        min_length : int, defaults to 0
            If > 0, retrieves only those representations
            that have at least `min_length` leaves
            in their tree.     
        """
        check_bracket_format(brackets)
        
        super().__init__(filename, strip_whitespace, pre_line_reader) # type: ignore
                                                                      # Types are correct here...

        self.detokenize : bool = detokenize
        """Whether to detokenize the trees after
        reading them from the file."""

        self.brackets : str = brackets
        """Bracket format of the source file. The first character is 
        an opening bracket. The second character is a closing bracket."""

        self.remove_non_words : bool = remove_non_words
        """Whether to remove all non-word leaves
        and non-word-only subtrees."""

        self.remove_traces : bool = remove_traces
        """Whether to remove all trace leaves
        and trace-only subtrees."""

        self.min_length : int = min_length
        """Minimum leaf count that a tree should
        have in order to be retrieved."""
    
    def __iter__(self) -> Iterator[PTBTreeRepresentation]:
        """Iterate through the file
        given at initialisation.
        Automatically detokenizes
        if `self.detokenize` is True.

        Yields
        ------
        Iterator[PTBTreeRepresentation]
            A tree representation.
        """

        for line in self.pre_line_reader:

            if len(line) > 0:
                
                ptb_tree : PTBTreeRepresentation
                ptb_tree = PTBTreeRepresentation.from_string(line, brackets = self.brackets)

                if self.detokenize:
                    try:
                        ptb_tree = ptb_tree.detokenize(inplace = True,
                                                       remove_non_words = self.remove_non_words,
                                                       remove_traces = self.remove_traces)
                        
                    # For a small fraction of trees detokenization is not possible.
                    except TreeTransformError:
                        continue
                
                else:
                    # if not detokenize, then remove non-words or traces manually
                    if self.remove_non_words:

                        # includes traces
                        ptb_tree = ptb_tree.remove_non_words(inplace = True)
                    
                    elif self.remove_traces:
                        # removes only traces but not punctuation
                        ptb_tree = ptb_tree.remove_traces(inplace = True)

                if self.min_length == 0:
                    yield ptb_tree

                elif len(ptb_tree.leaves) >= self.min_length:
                    yield ptb_tree


class ListTreeReader(Reader[TreeRepresentation]):
    """Reader for unlabelled bracketed
    trees. The file should contain one
    bracketed unlabelled tree per line.
    Produces `TreeRepresentation`s with
    dummy labels.

    Attributes
    ----------
    pre_json_reader : PreJSONReader
        Iterable that outputs `ListTree`
        objects.
    brackets : str
        Brackets to temporarily use when
        converting formats. 
    min_length : int
        Minimum number of leaves for
        trees for retrieval.
    """
    @overload
    def __init__(self, filename : str,
                 key : str,
                 pre_json_reader : None = None,
                 brackets : str = "[]",
                 min_length : int = 0):
        ...

    @overload
    def __init__(self, filename : None,
                 key : None,
                 pre_json_reader : PreJSONReader,
                 brackets : str = "[]",
                 min_length : int = 0):
        ...
        
    def __init__(self, filename : Optional[str] = None, 
                 key : Optional[str] = None,
                 pre_json_reader : Optional[PreJSONReader] = None,
                 brackets : str = "[]",
                 min_length : int = 0) -> None:
        """Initialises a ListTreeReader reader.
        One can specify a custom `PreJSONReader`
        or provide a path to a JSONL file and
        a key that represents the attribute that contains
        the ListTree.

        Parameters
        ----------
        filename : str, defaults to None
            The path to the file. Must be None
            if pre_json_reader is provided.
        key : str, defaults to True
            Attribute in the JSONL file that
            contains the `ListTree`. Must be None
            if pre_json_reader is provided.
        pre_json_reader : PreJSONReader, defaults to None
            Iterable that outputs `ListTree`
            objects. Must be None if filename and
            key are given.
        brackets : str, defaults to '[]'
            Brackets to temporarily use when
            converting formats. Must not occur
            in the leaves in the source file.
        min_length : int, defaults to 0
            If > 0, retrieves only those representations
            that have at least `min_length` leaves
            in their tree.     
        """
        
        check_bracket_format(brackets)
        
        self.pre_json_reader : PreJSONReader
        """Iterable that outputs `ListTree`
        objects."""

        if pre_json_reader is None:
            assert isinstance(filename, str) and isinstance(key, str)
            self.pre_json_reader = PreJSONReader(filename, key)
        
        else:
            assert filename is None and key is None
            self.pre_json_reader = pre_json_reader

        self.brackets : str = brackets
        """Brackets to temporarily use when
        converting formats. Must not occur
        in the leaves in the source file."""

        self.min_length : int = min_length
        """Minimum number of leaves for trees
        for retrieval."""

    def __iter__(self) -> Iterator[TreeRepresentation]:
        """Iterate through the file
        given at initialisation.

        Yields
        ------
        Iterator[TreeRepresentation]
            A tree representation.
        """

        listtree : ListTree
        for listtree in self.pre_json_reader:

            # Convert ListTree to bracketed string with dummy labels    
            ptb_format : str = self.get_ptb_format_from_diora_tree(listtree,
                                                                   brackets = self.brackets)
            ptb_tree : TreeRepresentation
            ptb_tree = TreeRepresentation.from_string(ptb_format, brackets = self.brackets)

            if self.min_length == 0:
                yield ptb_tree
            
            elif len(ptb_tree.leaves) >= self.min_length:
                yield ptb_tree

    @staticmethod
    def get_ptb_format_from_diora_tree(parse : ListTree,
                                       brackets : str = "[]",
                                       internal_label : str = GENERIC_NODE,
                                       leaf_label : str = GENERIC_LEAF) -> str:
        """Converts `ListTree` object to
        bracketed tree string with dummy labels.

        Parameters
        ----------
        parse : ListTree
            The tree to convert.
        brackets : str, defaults to '[]'
            The bracket format to use.
        internal_label : str, defaults to GENERIC_NODE
            Dummy label to use for internal
            nodes.
        leaf_label : str, defaults to GENERIC_LEAF
            Dummy label to use for
            pre-terminal nodes.

        Returns
        -------
        str
            Converted tree.
        """
        
        check_bracket_format(brackets)

        def _recursive_parser(parse : ListTree | str):
            if isinstance(parse, str):
                return f"{brackets[0]}{leaf_label} {parse}{brackets[1]}"
            
            else:
                return f"{brackets[0]}{internal_label} {' '.join(_recursive_parser(s) for s in parse)}{brackets[1]}"
            
        return _recursive_parser(parse)

class PTBReaderOptions(TypedDict, total = False):
    """Options for the initialisation
    of PTBReaders."""
    detokenize : bool
    remove_non_words : bool
    remove_traces : bool
    brackets : str

class ListTreeReaderOptions(TypedDict):
    """Options for the initialisation
    of ListTreeReader."""
    key : str
    brackets : NotRequired[str]  # Temporary brackets

class FlatReaderOptions(TypedDict):
    """Options for the initialisation
    of `FlatReaer`."""
    pass

@overload
def construct_reader(filetype : Literal["ptb"], filename : str, options : PTBReaderOptions,
                     min_length : int = 0) -> PTBReader:
    ...

@overload
def construct_reader(filetype : Literal["flat"], filename : str, options : FlatReaderOptions,
                     min_length : int = 0) -> FlatReader:
    ...

@overload
def construct_reader(filetype : Literal["json"], filename : str, options : ListTreeReaderOptions,
                     min_length : int = 0) -> ListTreeReader:
    ...

def construct_reader(filetype : FileType, 
                     filename : str, 
                     options : Union[PTBReaderOptions, 
                                     ListTreeReaderOptions, 
                                     FlatReaderOptions],
                     min_length : int = 0) -> Union[FlatReader, PTBReader, ListTreeReader]:
    """Construct an object based on a `Reader`
    subclass that fits with the requested filetype.

    Parameters
    ----------
    filetype : Literal['ptb', 'flat', 'json']
        Filetype to use.
        - 'ptb': PennTreebank (PTB), one tree per line
        - 'flat': simple string, one sequence per line
    filename : str
        Path to the file.
    options : ListTreeReaderOptions | FlatReaderOptions |
                PTBReaderOptions
        Options dictionary for the reader type.
    min_length : int, defaults to 0
        Only include sentences of minimum this
        length. Length is identified using
        whitespace between tokens.

    Returns
    -------
    Union[FlatReader, PTBReader, ListTreeReader]
        Reader object dependent on
        filetype.

    Raises
    ------
    ValueError
        If given an unknown filetype, it raises
        an exception.
    """

    match filetype:
        case "flat":
            return FlatReader(filename, strip_whitespace = True,
                              **options,
                              min_length = min_length)

        case "ptb":
            assert options is not None
            return PTBReader(filename, strip_whitespace = True,
                             **options,
                             min_length = min_length)
        
        case "json":
            assert options is not None
            options = cast(ListTreeReaderOptions, options)
            if "brackets" in options.keys():        # This looks ugly but the typechecker sees a problem
                                                    # when giving **{k : v for k, v in options.items() if k != "brackets"}
                                                    # as kwargs.
                return ListTreeReader(filename = None,  
                                      key = None,
                                      pre_json_reader = PreJSONReader(filename, key = options["key"]),
                                      brackets = options["brackets"],
                                      min_length = min_length)
            else:
                return ListTreeReader(filename = None,
                                      key = None,
                                      pre_json_reader = PreJSONReader(filename, key = options["key"]),
                                      min_length = min_length)
    
        case _:
            raise ValueError(f"File type {filetype} not allowed.")