"""Binarise a tree in PTB notation. Non-binary nodes are converted 
in a right-branching fashion.
"""

from diora.data.readers import construct_reader, PTBReaderOptions, PTBReader
from diora.data.representations import Tree, write_trees, PTBTreeRepresentation

import sys

from typing import List


def convert_to_binary(tree : Tree) -> Tree:
    if len(tree) > 2:
        new_right : Tree | str = tree.pop()
        new_left : Tree | str = tree.pop()

        new_start_idx : int = tree.start_idx if isinstance(new_left, str) else new_left.start_idx

        new_node : tree = Tree(label = "S", children = [new_left, new_right], start_idx = new_start_idx)

        tree.append(new_node)

        convert_to_binary(tree)
    
    else:
        for child in tree:
            if not isinstance(child, str):
                convert_to_binary(child)


if __name__ == '__main__':
    sourcefile : str = sys.argv[1]
    newfile : str = sys.argv[2]

    reader : PTBReader
    reader = construct_reader("ptb", 
                              sourcefile, 
                              PTBReaderOptions(detokenize = False,
                                               remove_non_words = False,
                                               remove_traces = False,
                                               brackets = "[]"),
                              min_length = 0)
    
    trees : List[PTBTreeRepresentation] = list(reader)
    for tree in trees:
        convert_to_binary(tree.tree)

    write_trees([tree.tree for tree in trees], newfile, "[]")