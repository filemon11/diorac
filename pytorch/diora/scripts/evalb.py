import collections
import json
import nltk
import os

from typing import Tuple, List, Union, Optional

Tree = List[Union[str, "Tree"]]

word_tags = set(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
               'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
               'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
               'WDT', 'WP', 'WP$', 'WRB'])

punctuation = set(['.', ',', ':', 'RRB', 'RCB', '\'\'', '...', '"', '?', '!', "'", ")", "--", 
                   '``', '`', 'LRB', 'LCB', "''", "'", "("])
punctuation = punctuation | set(["Ġ" + punct for punct in punctuation])
extra_tags = set(["<s>", "</s>", "<f>"])

def make_rb(tree : nltk.Tree) -> Tree:
    leaves = [nltk.Tree("DT", [x[0]]) for x in tree.pos()]
    
    current_tree = nltk.Tree("S", [])
    root = current_tree
    for i in leaves:
        current_tree.append(i)
        new_tree = nltk.Tree("S", [])
        current_tree.append(new_tree)
        current_tree = new_tree

    return root

def check_punctuation(to_check : str) -> bool:
    return (to_check in punctuation) or all([c in punctuation | {"Ġ"} for c in to_check])

def merge_subwords(tree : Tree, last_leaf : Union[None, Tree]) -> Tuple[bool, Optional[Tree]]:
    if len(tree) == 1 and isinstance(tree[0], str):
        if tree[0][0] != "Ġ" and not check_punctuation(tree[0]):
            if last_leaf is None:
                return False, tree
            
            else:
                assert len(last_leaf) > 0 and isinstance(last_leaf[0], str)
                last_leaf[0] = last_leaf[0] + tree[0]
                return True, last_leaf
            
        else:
            return False, tree
        
    else:
        to_del : List[bool] = []
        for child in tree:
            assert not isinstance(child, str)
            c_del, last_leaf = merge_subwords(child, last_leaf)

            to_del.append(c_del)

        for i, c_del in reversed(list(enumerate(to_del))):
            if c_del:
                del tree[i]

        if len(tree) == 0:
            return True, last_leaf
        
        else:
            return False, last_leaf
        
def replace_brackets(string : str) -> str:
    return string.replace("(", "-LRB-").replace(")", "-RRB-")

def get_ptb_format_from_nltk_tree(tr, force_dummy_labels = False):
    def helper(tr):
        if len(tr) == 1 and isinstance(tr[0], str):
            label = "DT" if force_dummy_labels else tr.label()
            return f'({label} {replace_brackets(tr[0])})'

        nodes = [helper(x) for x in tr]

        label = "S" if force_dummy_labels else tr.label()
        return f'({label} {" ".join(nodes)})'

    out = helper(tr)

    if out.startswith('(DT'):
        out = f'(S {out})'

    return out


def get_ptb_format_from_diora_tree(parse, tokens, return_string=False):

    def recursive_add_tokens(parse):
        def helper(tr, pos):
            if not isinstance(tr, (tuple, list)):
                return 1, tokens[pos]

            size, nodes = 0, []
            for x in tr:
                xsize, xnode = helper(x, pos + size)
                size += xsize
                nodes.append(xnode)

            return size, tuple(nodes)

        _, new_parse = helper(parse, 0)

        return new_parse

    def recursive_string(parse):
        if isinstance(parse, str):
            return f'[DT {parse}]'
        return '[S ' + ' '.join([recursive_string(p) for p in parse]) + ']'

    parse = recursive_add_tokens(parse)
    if return_string:
        parse = recursive_string(parse)
    return parse


def remove_punctuation_from_tree(tree, ref, punct):
    def recursive_remove_using_mask(tr, position, mask) -> Tuple[int, bool, object | None]:
        if len(tr) == 1 and isinstance(tr[0], str):
            size = 1
            keep = mask[position]
            return size, keep, None

        size = 0
        keep = []
        for i, x in enumerate(tr):
            xsize, xkeep, uptree = recursive_remove_using_mask(x, position + size, mask)
            size += xsize
            keep.append(xkeep)
            if uptree is not None:
                tr[i] = uptree

        for i, xkeep in list(enumerate(keep))[::-1]:
            if not xkeep:
                del tr[i]

        keep = any(keep)
        return size, keep, (tr[0] if len(tr) == 1 else None)

    tokens = tree.leaves()
    #part_of_speech = [x[1] for x in ref.pos()]
    words = [x[0] for x in ref.pos()]
    mask = [not x in punct and not all([c in punct | {"Ġ"} for c in x])
                for x in words] # Tokens that are punctuation are given False in mask.
    new_tokens = [x for x, m in zip(tokens, mask) if m]
    assert len(new_tokens) > 0

    recursive_remove_using_mask(tree, 0, mask)
    if len(tree) == 1 and not isinstance(tree[0], str):    # remove unary root node?
        tree = tree[0]

    assert len(tree.leaves()) == len(new_tokens), (tree.leaves(), new_tokens, tokens, mask)
    assert tuple(tree.leaves()) == tuple(new_tokens)

    return tree


def main(args):
    os.system(f'mkdir -p {args.out}')

    # Read DIORA data.
    pred = []
    with open(args.pred) as f:
        for line in f:
            tree = json.loads(line)['tree']
            pred.append(tree)

    # Read ground truth parse trees.
    num_problems : int = 0
    gold = []
    with open(args.gold) as f:
        for line in f:
            try:
                nltk_tree = nltk.Tree.fromstring(line,
                                                 brackets = "[]")
                gold.append(nltk_tree)
            except Exception as ex:
                print(line)
                print(ex)
                num_problems += 1

    print("Number of problems", num_problems)
    assert len(gold) == len(pred), f"The gold and pred files must have same number of sentences. {len(pred)} != {len(gold)}"

    print(len(gold[0].leaves()))
    def num_leaves(l):
        return sum([(num_leaves(d) if isinstance(d, list) else 1) for d in l])
    print(num_leaves(pred[0]))
    print(gold[0].leaves())
    print(pred[0])
    print(get_ptb_format_from_diora_tree(pred[0], gold[0].leaves(), return_string=True))
    #print(pred)
    pred = [nltk.Tree.fromstring(get_ptb_format_from_diora_tree(p, g.leaves(), return_string=True), brackets = "[]")
            for g, p in zip(gold, pred)]

    #pred = [make_rb(t) for t in pred]
    # Remove punctuation
    #pred = [remove_punctuation_from_tree(p, ref=g, punct = punctuation) for g, p in zip(gold, pred)]
    #gold = [remove_punctuation_from_tree(g, ref=g, punct = punctuation) for g, p in zip(gold, pred)]
    print(gold[3])
    
    # Remove extra tags
    pred = [remove_punctuation_from_tree(p, ref=g, punct = extra_tags) for g, p in zip(gold, pred)]
    gold = [remove_punctuation_from_tree(g, ref=g, punct = extra_tags) for g, p in zip(gold, pred)]
    print(gold[3])
    print(pred[3])
    #for g, p in zip(gold, pred):
    #    merge_subwords(g, None)
    #    merge_subwords(p, None)

    #print(pred[0])

    # Remove sentences according to length.
    assert all(len(x.leaves()) > 0 for x in pred)
    assert all(len(x.leaves()) > 0 for x in gold)
    assert all(len(p.leaves()) == len(g.leaves()) for g, p in zip(gold, pred))


    if args.max_length is not None:
        for i, p in reversed(list(enumerate(pred))):
            if num_leaves(p) > args.max_length:
                del pred[i]
                del gold[i]


    # Serialize as strings.
    pred = [get_ptb_format_from_nltk_tree(p) for g, p in zip(gold, pred)]
    # need to replace labels in gold tree since some of the newly introduced labels produce problems
    gold = [get_ptb_format_from_nltk_tree(g, force_dummy_labels = True) for g, p in zip(gold, pred)]

    # Write new intermediate files.
    new_pred_file = os.path.join(args.out, 'pred.txt')
    new_gold_file = os.path.join(args.out, 'gold.txt')

    with open(new_pred_file, 'w') as f:
        for parse in pred:
            f.write(parse + '\n')

    with open(new_gold_file, 'w') as f:
        for parse in gold:
            f.write(parse + '\n')

    # Run EVALB.
    evalb_exe = os.path.join(args.evalb, 'evalb')
    evalb_out_file = os.path.join(args.out, 'evalb.out')

    assert os.path.exists(evalb_exe), "Did not detect evalb executable. Try `(cd EVALB && make)`."
    assert os.path.exists(args.evalb_config)

    evalb_command = '{evalb} -p {evalb_config} {gold} {pred} > {out}'.format(
        evalb=os.path.join(args.evalb, 'evalb'),
        evalb_config=args.evalb_config,
        gold=new_gold_file,
        pred=new_pred_file,
        out=evalb_out_file)

    print(f'\nRunning: {evalb_command}')
    os.system(evalb_command)

    print(f'\nResults are ready at: {evalb_out_file}')

    print(f'\n==== PREVIEW OF RESULTS ({evalb_out_file}) ====\n')
    os.system(f'tail -n 27 {evalb_out_file}')
    print('')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='File with DIORA predictions from `parse.py`.')
    parser.add_argument('--gold', type=str, required=True, help='File with ground truth parse trees in PTB format.')
    parser.add_argument('--out', type=str, required=True, help='Directory to write intermediates files for EVALB and results.')
    parser.add_argument('--evalb', type=str, required=True, help='Path to EVALB directory.')
    parser.add_argument('--evalb_config', type=str, required=True, help='Path to EVALB configuration file.')
    parser.add_argument('--max_length', default=None, type=int, help='Max length after removing punctuation.')
    args = parser.parse_args()
    main(args)
