import collections
import json
import os
import types

import torch

from tqdm import tqdm

from train import argument_parser, parse_args, configure
from train import build_net, BuildNetOptions
from diora.data.build_corpus import prepare_test, IteratorOptions, LossOptions
from diora.data.representations import write_trees
from diora.logging.configuration import get_logger

from diora.analysis.cky import ParsePredictor as CKY


punctuation_words = set([x.lower() for x in ['.', ',', ':', '-LRB-', '-RRB-', '\'\'',
    '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']])


def remove_using_flat_mask(tr, mask):
    kept, removed = [], []
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            if mask[pos] == False:
                removed.append(tr)
                return None, 1
            kept.append(tr)
            return tr, 1

        size = 0
        node = []

        for subtree in tr:
            x, xsize = func(subtree, pos=pos + size)
            if x is not None:
                node.append(x)
            size += xsize

        if len(node) == 1:
            node = node[0]
        elif len(node) == 0:
            return None, size
        return node, size
    new_tree, _ = func(tr)
    return new_tree, kept, removed


def flatten_tree(tr):
    def func(tr):
        if not isinstance(tr, (list, tuple)):
            return [tr]
        result = []
        for x in tr:
            result += func(x)
        return result
    return func(tr)


def postprocess(tr, tokens=None):
    if tokens is None:
        tokens = flatten_tree(tr)

    # Don't remove the last token. It's not punctuation.
    if tokens[-1].lower() not in punctuation_words:
        return tr

    mask = [True] * (len(tokens) - 1) + [False]
    tr, kept, removed = remove_using_flat_mask(tr, mask)
    assert len(kept) == len(tokens) - 1, 'Incorrect tokens left. Original = {}, Output = {}, Kept = {}'.format(
        binary_tree, tr, kept)
    assert len(kept) > 0, 'No tokens left. Original = {}'.format(tokens)
    assert len(removed) == 1
    tr = (tr, tokens[-1])

    return tr


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


def replace_leaves(tree, leaves):
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            return 1, leaves[pos]

        newtree = []
        sofar = 0
        for node in tr:
            size, newnode = func(node, pos+sofar)
            sofar += size
            newtree += [newnode]

        return sofar, newtree

    _, newtree = func(tree)

    return newtree


class FileWriter(object):
    def __init__(self, output_file=None, retain_file_order=False, idx2word=None):
        self.output_file = output_file
        self.retain_file_order = retain_file_order
        self.idx2word = idx2word
        self.state = collections.defaultdict(list)

        if not self.retain_file_order:
            self.f = open(output_file, 'w')

    def update(self, batch_map, trees):
        idx2word = self.idx2word

        for ii, tr in enumerate(trees):
            example_id = batch_map['index'][ii]

            s = [idx2word[idx] for idx in batch_map['input_ids'][ii].tolist()]
            tr = replace_leaves(tr, s)
            if options.postprocess:
                tr = postprocess(tr, s)
            o = collections.OrderedDict(example_id=example_id, tree=tr)

            if not self.retain_file_order:
                self.f.write(json.dumps(o) + '\n')

            else:
                file_order = batch_map['index'][ii]
                self.state['file_order'].append(file_order)
                self.state['o'].append(o)

    def finish(self):
        if not self.retain_file_order:
            self.f.close()

        else:
            with open(self.output_file, 'w') as f:
                for idx, o in sorted(zip(self.state['file_order'], self.state['o']), key=lambda x: x[0]):
                    f.write(json.dumps(o) + '\n')


def run(options):
    logger = get_logger()

    #validation_dataset = get_validation_dataset(options)
    #validation_iterator = get_validation_iterator(options, validation_dataset)

    iterator_options = IteratorOptions(cuda = options.cuda,
                                       num_workers = options.num_workers,
                                       local_rank = options.local_rank,
                                       k_neg = options.k_neg,
                                       batch_size = options.batch_size,
                                       length_to_size = options.length_to_size)
    
    loss_options = LossOptions(freq_dist_power = options.freq_dist_power,
                               reconstruct_mode = options.reconstruct_mode)
    
    validation_trees, validation_iterator, word2i, char2i = prepare_test(
                                                                options.validation_data_type,
                                                                options.validation_path,
                                                                options.tokenizer,
                                                                options.word2i_path,
                                                                options.char2i_path,
                                                                iterator_options,
                                                                loss_options,
                                                                validation_filter_length = options.validation_filter_length)
    
    write_trees(validation_trees, options.transformed_validation_trees_path)
    #word2idx = validation_dataset['word2idx']
    #embeddings = validation_dataset['embeddings']

    i2word = {v: k for k, v in word2i.items()}

    logger.info('Initializing model.')

    build_net_options : BuildNetOptions = dict(
            lr = options.lr,
            hidden_dim = options.hidden_dim,
            k_neg = options.k_neg,
            margin = options.margin,
            normalize = options.normalize,
            cuda = options.cuda,
            multigpu = options.multigpu,
            ngpus = torch.cuda.device_count(),
            local_rank = options.local_rank,
            master_addr = options.master_addr,
            master_port = options.master_port,
            arch = options.arch,
            load_model_path = options.load_model_path,
            experiment_name = options.experiment_name,
            reconstruct_mode = options.reconstruct_mode,
            unk_idx = word2i.default_factory()
    )
    trainer = build_net(build_net_options, word2i, char2i)


    ###

    #trainer.prepare_pred()
    #batches = validation_iterator.get_iterator(random_seed=options.seed)
    #print("dataset_size:", validation_iterator.dataset_size)
    #
        #output_file = os.path.abspath(os.path.join(options.experiment_path, 'parse.jsonl'))
    #
        #logger.info('Beginning.')
        #logger.info('Writing output to = {}'.format(output_file))
    #
        #file_writer = FileWriter(output_file=output_file, retain_file_order=options.retain_file_order, idx2word=i2word)
    #
        #total_size : int = 0
        #with torch.no_grad():
        #    for i, batch_map in tqdm(enumerate(batches)):
    #
        #        trees = trainer.pred_step(batch_map)
    #
        #        file_writer.update(batch_map, trees)
    #
    #print("total_size:", total_size)
    #file_writer.finish()
    ###

    # Parse

    diora = trainer.net.diora

    ## Monkey patch parsing specific methods.
    override_init_with_batch(diora)
    override_inside_hook(diora)

    ## Turn off outside pass.
    trainer.net.diora.outside = False

    ## Eval mode.
    trainer.net.eval()

    ## Parse predictor.
    parse_predictor = CKY(net=diora)

    batches = validation_iterator.get_iterator(random_seed=options.seed)
    print("dataset_size:", validation_iterator.dataset_size)

    output_file = os.path.abspath(os.path.join(options.experiment_path, 'parse.jsonl'))

    logger.info('Beginning.')
    logger.info('Writing output to = {}'.format(output_file))

    file_writer = FileWriter(output_file=output_file, retain_file_order=options.retain_file_order, idx2word=i2word)

    total_size : int = 0
    with torch.no_grad():
        for i, batch_map in tqdm(enumerate(batches)):
            sentences = batch_map['input_ids']
            batch_size = sentences.shape[0]
            total_size += batch_size
            length = sentences.shape[1]

            # Skip very short sentences.
            if length <= 2:
                if length == 2:
                    trees = [(0, 1) for _ in range(batch_size)]
                elif length == 1:
                    trees = [0 for _ in range(batch_size)]
                else:
                    raise ValueError

                file_writer.update(batch_map, trees)
                continue

            _ = trainer.step(batch_map, train=False, compute_loss=False)

            trees = parse_predictor.parse_batch(batch_map)

            file_writer.update(batch_map, trees)

    print("total_size:", total_size)
    file_writer.finish()


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
