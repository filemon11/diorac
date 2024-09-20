import argparse
import datetime
import math
import os
import random
import sys
import uuid

import torch
import torch.nn as nn
import torch.backends.cuda

import optuna
from optuna.trial import TrialState

from diora.data.build_corpus import prepare_train_val, IteratorOptions, LossOptions
from diora.data.batch_iterator import BatchIterator, Char2i, BatchMap
from diora.data.tokenizer import Word2i

from diora.net.trainer import build_net as trainer_build_net
from diora.net.trainer import BuildNetOptions, Trainer

from diora.utils.path import package_path
from diora.logging.configuration import configure_experiment, get_logger
from diora.utils.flags import stringify_flags, init_with_flags_file, save_flags
from diora.utils.checkpoint import save_experiment

from diora.net.experiment_logger import ExperimentLogger

from diora.scripts.argtypes import Args

from torch.profiler import schedule, profile, record_function, ProfilerActivity

import logging

from copy import copy

from typing import Optional, Dict, Callable, Iterator

data_types_choices = ('nli', 'conll_jsonl', 'flat', 'flat_id', 'synthetic', 'jsonl', 'ptb')

torch.backends.cuda.matmul.allow_tf32 = True
print("Device name " + str(torch.cuda.get_device_name(torch.cuda.current_device())))
print("Device " + str(torch.cuda.current_device()))

def print_best_trial_so_far(study, trial):
    logger = get_logger()

    logger.info('\nBest trial so far used the following hyper-parameters:')
    for key, value in study.best_trial.params.items():
        print('{}: {}'.format(key, value))
    logger.info('to achieve objective function score of {}\n'.format(study.best_trial.value))


def count_params(net):
    return sum([x.numel() for x in net.parameters() if x.requires_grad])


def build_net(options : BuildNetOptions) -> Trainer:

    trainer = trainer_build_net(options)

    logger = get_logger()
    logger.info('# of params = {}'.format(count_params(trainer.net)))

    return trainer


def generate_seeds(n, seed=11):
    random.seed(seed)
    for _ in range(n):
        yield random.randint(0, 2**16)

def getmyiterator(seed : int) -> Callable[[BatchIterator, Optional[int]], Iterator[tuple[int, BatchMap]]]:
    def myiterator(iterator : BatchIterator, 
                   stop_after : int | None = None) -> Iterator[tuple[int, BatchMap]]:
        it = iterator.get_iterator(random_seed = seed)
        count = 0
        for batch_map in it:
            # TODO: Skip short examples (optionally).
            if batch_map['length'] <= 2:
                continue
            yield count, batch_map
            count += 1

            if stop_after is not None and count >= stop_after:
                break

    return myiterator

def check_accum_do_step(batch_idx : int, accum_iter : int, 
                        batch_size : int, dataset_size : int) -> bool:
    is_accumulate_final : bool = (batch_idx + 1) % accum_iter == 0
    is_last_batch : bool = batch_idx + 1 == dataset_size // batch_size
    
    return is_accumulate_final or is_last_batch

def train_step(trainer: Trainer, batch_map : BatchMap,
               batch_idx : int, accum_iter : int, 
               batch_size : int, dataset_size : int) -> torch.Tensor:
    accum_do_step : bool = check_accum_do_step(batch_idx, 
                                               accum_iter, 
                                               batch_size,
                                               dataset_size)
            
    return trainer.step(batch_map, do_step = accum_do_step)

def validate(myiterator : Iterator[tuple[int, BatchMap]], trainer : Trainer) -> float:
    num_batches : int = 0
    val_result : float = 0
    for _, val_batch_map in myiterator:
        val_result += trainer.step(val_batch_map, train = False)["total_loss"]
        num_batches += 1
        
    return val_result / num_batches
    
def train_hyperopt(trial, 
                   train_iterator : BatchIterator, 
                   validation_iterator : BatchIterator, 
                   trainer : Trainer) -> float:
    """Returns validation loss"""
    logger = get_logger()
    
    seeds_train = generate_seeds(options.max_epoch, options.seed)
    seeds_val = generate_seeds(options.max_epoch, options.seed+1)

    step = 0

    accum_iter : int = options.accum_iter

    for epoch, seed_train, seed_val in zip(range(options.max_epoch), seeds_train, seeds_val):
        # --- Train--- #

        myiterator_train = getmyiterator(seed_train)
        myiterator_val = getmyiterator(seed_val)

        logger.info("Device " + str(torch.cuda.current_device()))

        for batch_idx, batch_map in myiterator_train(train_iterator, None):
            logger.info(f"Epoch {epoch}, Sentence {batch_idx * train_iterator.config.get('batch_size')}/{len(train_iterator.sentences['input_ids'])}")
            
            train_step(trainer, batch_map, batch_idx,
                       accum_iter, options.batch_size,
                       train_iterator.dataset_size)

            val_result = validate(myiterator_val(validation_iterator, 20), trainer)
            logger.info(f"Validation loss: {val_result}")
            trial.report(val_result, step)
            
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            step += 1

        if options.max_step is not None and step >= options.max_step:
            break
    
    return val_result

def run_train(options, train_iterator : BatchIterator, trainer : Trainer, validation_iterator : BatchIterator | None):
    logger = get_logger()
    experiment_logger = ExperimentLogger()

    logger.info('Running train.')

    seeds = generate_seeds(options.max_epoch, options.seed)

    step = 0

    checks_without_improvement : int = 0
    best_val : float = math.inf

    accum_iter : int = options.accum_iter

    #def trace_handler(p):
    #    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    #    #print(output)
    #    p.export_chrome_trace(os.path.join(options.experiment_path, "trace_" + str(p.step_num) + ".json"))

    #my_schedule = schedule(
    #    skip_first = 10,
    #    wait = 5,
    #    warmup = 1,
    #    active = 3
    #)

    #with profile(
    #    activities = [ProfilerActivity.CUDA],
    #    schedule = my_schedule,
    #    on_trace_ready = trace_handler
    #) as pr:
    if True:
        for epoch, seed in zip(range(options.max_epoch), seeds):
            # --- Train--- #

            logger.info('epoch={} seed={}'.format(epoch, seed))

            num_val_batches = 20
            myiterator = getmyiterator(seed)

            for batch_idx, batch_map in myiterator(train_iterator, None):
                if options.finetune and step >= options.finetune_after:
                    trainer.freeze_diora()

                #print(batch_map["input_ids"].device, options.local_rank)
                result = train_step(trainer, batch_map, batch_idx,
                                    accum_iter, options.batch_size,
                                    train_iterator.dataset_size)

                experiment_logger.record(result)

                if step % options.log_every_batch == 0:
                    experiment_logger.log_batch(epoch, step, batch_idx, batch_size = options.batch_size)

                # -- Periodic Checkpoints -- #

                if not options.multigpu or options.local_rank == 0:
                    if step % options.save_latest == 0 and step >= options.save_after:
                        logger.info('Saving model (periodic).')
                        trainer.save_model(os.path.join(options.experiment_path, 'model_periodic.pt'))
                        save_experiment(os.path.join(options.experiment_path, 'experiment_periodic.json'), step)

                        if validation_iterator is not None:
                            val_result = validate(myiterator(validation_iterator, num_val_batches), trainer)
                            logger.info(f"Validation loss: {val_result}")

                            if val_result < best_val:
                                best_val = val_result

                            else:
                                checks_without_improvement += 1

                            if False and checks_without_improvement > options.early_stopping:
                                logger.info('Early stopping. No improvement since {} steps.'.format(options.save_latest*checks_without_improvement))
                                sys.exit()
#
                    if step % options.save_distinct == 0 and step >= options.save_after:
                        logger.info('Saving model (distinct).')
                        trainer.save_model(os.path.join(options.experiment_path, 'model.step_{}.pt'.format(step)))
                        save_experiment(os.path.join(options.experiment_path, 'experiment.step_{}.json'.format(step)), step)

                del result

                step += 1
                #pr.step()

            experiment_logger.log_epoch(epoch, step)

            if options.max_step is not None and step >= options.max_step:
                logger.info('Max-Step={} Quitting.'.format(options.max_step))
                sys.exit()

        torch.distributed.destroy_process_group()

class Objective:
    def __init__(self, configs: dict[str, tuple[BatchIterator, BatchIterator, BuildNetOptions]], group):
        self.configs : dict[str, tuple[BatchIterator, BatchIterator, BuildNetOptions]] = configs
        self.group = group

    def __call__(self, trial) -> float:
        logger = get_logger()

        trial = optuna.integration.TorchDistributedTrial(trial, self.group)

        tokenizer_name = trial.suggest_categorical("tokenizer", list(self.configs.keys()))
        
        train_iterator, validation_iterator, build_net_options = self.configs[tokenizer_name]

        build_net_options : BuildNetOptions = copy(build_net_options)
        build_net_options["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log = True)
        build_net_options["hidden_dim"] = trial.suggest_int("hidden_dim", 200, 1000, log = True)
        build_net_options["inner_dim"] = trial.suggest_int("inner_dim", 200, 2400, log = True)
        build_net_options["arch"] = trial.suggest_categorical("arch", ["mlp", "mlp-shared", "treelstm"])
        build_net_options["reconstruct_mode"] = trial.suggest_categorical("reconstruct_mode", ["margin", "softmax"])
        # inner dim

        logger.info("Params: tokenizer = {}, lr = {}, hidden_dim = {}, inner_dim = {}, arch = {}, reconstruction_mode = {}".format(
                                                                                    tokenizer_name,
                                                                                    build_net_options["lr"],
                                                                                    build_net_options["hidden_dim"],
                                                                                    build_net_options["inner_dim"],
                                                                                    build_net_options["arch"],
                                                                                    build_net_options["reconstruct_mode"]))
    
        trainer = build_net(build_net_options)
        
        loss : float = train_hyperopt(trial, train_iterator,
                                      validation_iterator, trainer)
        
        return loss

def prepare(options : Args, ngpus : int) -> tuple[BatchIterator, BatchIterator, BuildNetOptions]:

    iterator_options = IteratorOptions(cuda = options.cuda,
                                       ngpus = ngpus,
                                       num_workers = options.num_workers,
                                       local_rank = options.local_rank,
                                       k_neg = options.k_neg,                   # fixed HP 
                                       batch_size = options.batch_size,
                                       length_to_size = options.length_to_size) # fixed HP

    loss_options = LossOptions(freq_dist_power = options.freq_dist_power)
    
    train_iterator, validation_iterator, word2i, char2i = prepare_train_val(
                                            options.train_data_type,
                                            options.train_path,
                                            options.validation_data_type,
                                            options.validation_path,
                                            options.tokenizer,
                                            options.word2i_path,
                                            options.char2i_path,
                                            iterator_options,
                                            loss_options,
                                            options.train_filter_length,        # fixed HP (more should be better)
                                            options.validation_filter_length,
                                            embeddings_type = "word",
                                            fragments = options.fragments,
                                            add_future = options.add_future,
                                            use_validation = options.use_validation)

    build_net_options : BuildNetOptions = dict(
            lr = options.lr,                    # HP
            hidden_dim = options.hidden_dim,    # HP
            inner_dim = options.inner_dim,      # HP
            k_neg = options.k_neg,              # fixed HP
            margin = options.margin,
            normalize = options.normalize,      # fixed HP
            cuda = options.cuda,
            ngpus = ngpus,
            multigpu = options.multigpu,
            local_rank = options.local_rank,
            master_addr = options.master_addr,
            master_port = options.master_port,
            arch = options.arch,                # HP
            load_model_path = options.load_model_path,
            experiment_name = options.experiment_name,
            reconstruct_mode = options.reconstruct_mode,    # HP
            unk_idx = word2i.default_factory(),
            word_embed_len = len(word2i),
            char_embed_len = 0 if char2i is None else len(char2i)
    )

    return train_iterator, validation_iterator, build_net_options


def run(options : Args):
    logger : logging.Logger = get_logger()

    ngpus : int = 1
    local_rank : int = 0
    if options.cuda and options.multigpu:

        ngpus = torch.cuda.device_count()

        # initialize the process group
        gr = torch.distributed.init_process_group('nccl')

        local_rank = torch.distributed.get_rank()

        torch.cuda.set_device(local_rank)

        logger.info(f"Start running on rank {local_rank}.")
        options.local_rank = local_rank

        logger.info("Device name " + str(torch.cuda.get_device_name(torch.cuda.current_device())))
        logger.info("Device " + str(torch.cuda.current_device()))

    logger.info('Initializing model.')

    if options.hyperopt:
        
        tokenizer : dict[str, str] = {"16k" : os.path.expanduser("~/Diora/dioralm/pytorch/diora/babylm_bpe_tokenizer_16k"),
                                      "32k" : os.path.expanduser("~/Diora/dioralm/pytorch/diora/babylm_bpe_tokenizer_32k")}
        
        configs : dict[str, tuple[BatchIterator, BatchIterator, BuildNetOptions]] = {}

        for tokenizer_name, tokenizer_path in tokenizer.items():
            _options = copy(options)
            _options.tokenizer = tokenizer_path
            configs[tokenizer_name] = prepare(_options, ngpus)

        objective : Objective = Objective(configs, gr)

        if local_rank == 0:
            study = optuna.create_study(direction = "minimize",
                                        sampler = optuna.samplers.TPESampler(seed = options.seed),
                                        pruner = optuna.pruners.HyperbandPruner(min_resource = 3, reduction_factor = 3))
            study.optimize(objective, n_trials = options.n_trials, callbacks=[print_best_trial_so_far])

        else:
            for _ in range(options.n_trials):
                try:
                    objective(None)
                except optuna.TrialPruned:
                    pass
        
        if local_rank == 0:
            assert study is not None
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            logger.info("Study statistics: ")
            logger.info("  Number of finished trials: " + str(len(study.trials)))
            logger.info("  Number of pruned trials: " + str(len(pruned_trials)))
            logger.info("  Number of complete trials: " + str(len(complete_trials)))

            logger.info("Best trial:")
            trial = study.best_trial

            logger.info("  Value: " + str(trial.value))
            logger.info("  Params: ")
            for key, value in trial.params.items():
                logger.info("    {}: {}".format(key, value))

    else:

        train_iterator, validation_iterator, build_net_options = prepare(options, ngpus)

        trainer = build_net(build_net_options)
        logger.info('Model:')
        for name, p in trainer.net.named_parameters():
            logger.info('{} {}'.format(name, p.shape))

        if options.save_init:
            logger.info('Saving model (init).')
            trainer.save_model(os.path.join(options.experiment_path, 'model_init.pt'))

        run_train(options, train_iterator, trainer, validation_iterator)


def argument_parser():
    parser = argparse.ArgumentParser()

    # Debug.
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--git_sha', default=None, type=str)
    parser.add_argument('--git_branch_name', default=None, type=str)
    parser.add_argument('--git_dirty', default=None, type=str)
    parser.add_argument('--uuid', default=None, type=str)
    parser.add_argument('--model_flags', default=None, type=str,
                        help='Load model settings from a flags file.')
    parser.add_argument('--flags', default=None, type=str,
                        help='Load any settings from a flags file.')

    parser.add_argument('--master_addr', default='127.0.0.1', type=str)
    parser.add_argument('--master_port', default='29500', type=str)
    parser.add_argument('--world_size', default=None, type=int)

    # Pytorch
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--multigpu', action='store_true')
    parser.add_argument("--local_rank", default=None, type=int) # for distributed-data-parallel
    parser.add_argument("--num_workers", default=0, type=int) # for batch iterator

    # Logging.
    parser.add_argument('--default_experiment_directory', default=os.path.join(package_path(), '..', 'log'), type=str)
    parser.add_argument('--experiment_name', default=None, type=str)
    parser.add_argument('--experiment_path', default=None, type=str)
    parser.add_argument('--log_every_batch', default=10, type=int)
    parser.add_argument('--save_latest', default=1000, type=int)
    parser.add_argument('--save_distinct', default=50000, type=int)
    parser.add_argument('--save_after', default=1000, type=int)
    parser.add_argument('--save_init', action='store_true')

    # Loading.
    parser.add_argument('--load_model_path', default=None, type=str)

    # Data.
    parser.add_argument('--data_type', default='nli', choices=data_types_choices)
    parser.add_argument('--train_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--validation_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--train_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_train.jsonl'), type=str)
    parser.add_argument('--validation_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_dev.jsonl'), type=str)
    parser.add_argument('--transformed_validation_trees_path', default=None, type=str)
    parser.add_argument('--embeddings_path', default=os.path.expanduser('~/data/glove/glove.6B.300d.txt'), type=str)
    parser.add_argument('--tokenizer', default=os.path.expanduser("~/Diora/dioralm/pytorch/diora/babylm_bpe_tokenizer_16k"), type=str)
    parser.add_argument('--word2i_path', default=None, type=str)
    parser.add_argument('--w2i_from_tokenizer', action='store_true')
    parser.add_argument("--all_from_w2i", action='store_true')

    # Data (synthetic).
    parser.add_argument('--synthetic-nexamples', default=1000, type=int)
    parser.add_argument('--synthetic-vocabsize', default=1000, type=int)
    parser.add_argument('--synthetic-embeddingsize', default=1024, type=int)
    parser.add_argument('--synthetic-minlen', default=20, type=int)
    parser.add_argument('--synthetic-maxlen', default=21, type=int)
    parser.add_argument('--synthetic-seed', default=11, type=int)
    parser.add_argument('--synthetic-length', default=None, type=int)
    parser.add_argument('--use-synthetic-embeddings', action='store_true')

    # Data (preprocessing).
    parser.add_argument('--uppercase', action='store_true')
    parser.add_argument('--train_filter_length', default=None, type=int)
    parser.add_argument('--validation_filter_length', default=None, type=int)

    # Model.
    parser.add_argument('--arch', default='treelstm', choices=('treelstm', 'mlp', 'mlp-shared'))
    parser.add_argument('--hidden_dim', default=10, type=int)
    parser.add_argument('--inner_dim', default=None, type=int, help="Hidden dimensionality of combiner networks; if None: equal to hidden_dim parameter")
    parser.add_argument('--normalize', default='unit', choices=('none', 'unit'))
    parser.add_argument('--compress', action='store_true',
                        help='If true, then copy root from inside chart for outside. ' + \
                             'Otherwise, learn outside root as bias.')

    # Model (Objective).
    parser.add_argument('--reconstruct_mode', default='margin', choices=('margin', 'softmax'))

    # Model (Embeddings).
    parser.add_argument('--emb', default='w2v', choices=('w2v', 'elmo', 'both'))

    # Model (Negative Sampler).
    parser.add_argument('--margin', default=1, type=float)
    parser.add_argument('--k_neg', default=3, type=int)
    parser.add_argument('--freq_dist_power', default=0.75, type=float)

    # Fragments
    parser.add_argument('--fragments', default=False, choices=('random', 'all'), help='Mode for inclusion of fragments.')
    parser.add_argument('--add_future', default=False, help='Whether to add future <f> tokens. ATTENTION: Need to change loss functions to disregard last token in loss.')

    # ELMo
    parser.add_argument('--elmo_options_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', type=str)
    parser.add_argument('--elmo_weights_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', type=str)
    parser.add_argument('--elmo_cache_dir', default=None, type=str,
                        help='If set, then context-insensitive word embeddings will be cached ' + \
                             '(identified by a hash of the vocabulary).')

    # Training.
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--length_to_size', default=None, type=str,
                        help='Easily specify a mapping of length to batch_size.' + \
                             'For instance, 10:32,20:16 means that all batches' + \
                             'of length 10-19 will have batch size 32, 20 or greater' + \
                             'will have batch size 16, and less than 10 will have batch size' + \
                             'equal to the batch_size arg. Only applies to training.')
    parser.add_argument('--train_dataset_size', default=None, type=int)
    parser.add_argument('--validation_dataset_size', default=None, type=int)
    parser.add_argument('--validation_batch_size', default=None, type=int)
    parser.add_argument('--use_validation', action='store_true', help='Report validation score during training')
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--max_step', default=None, type=int)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_after', default=0, type=int)
    parser.add_argument('--early_stopping', default=80, type=int, help='Maximum early stopping checks.')
    parser.add_argument('--accum_iter', default=4, type=int, help="How many batches to accumulate before performing gradient descent.")

    # Hyperparameter optimisation.
    parser.add_argument('--hyperopt', action = 'store_true', help = "Whether to perform hyperparameter optimisation.")
    parser.add_argument('--n_trials', default = 100, help = "Number of trials to perform for hyperparameter optimisation.")

    # Parsing.
    parser.add_argument('--postprocess', action='store_true')
    parser.add_argument('--retain_file_order', action='store_true') # If true, then outputs are written in same order as read from file.

    # Optimization.
    parser.add_argument('--lr', default=4e-3, type=float)

    return parser


def parse_args(parser : argparse.ArgumentParser) -> Args:
    options = Args(**vars(parser.parse_args()))

    # Set default flag values (data).
    options.train_data_type = options.data_type if options.train_data_type is None else options.train_data_type
    options.validation_data_type = options.data_type if options.validation_data_type is None else options.validation_data_type
    options.validation_batch_size = options.batch_size if options.validation_batch_size is None else options.validation_batch_size
    options.validation_filter_length = options.train_filter_length if options.validation_filter_length == 0 else options.validation_filter_length
    # Set default flag values (config).

    if options.transformed_validation_trees_path is None:
        options.transformed_validation_trees_path = os.path.join(os.path.dirname(options.validation_path), f"{options.experiment_name}_transformed.{options.validation_data_type}")
    if not options.git_branch_name:
        options.git_branch_name = os.popen(
            'git rev-parse --abbrev-ref HEAD').read().strip()

    if not options.git_sha:
        options.git_sha = os.popen('git rev-parse HEAD').read().strip()

    if not options.git_dirty:
        options.git_dirty = os.popen("git diff --quiet && echo 'clean' || echo 'dirty'").read().strip()

    if not options.uuid:
        options.uuid = str(uuid.uuid4())

    if not options.experiment_name:
        options.experiment_name = '{}'.format(options.uuid[:8])

    if not options.experiment_path:
        options.experiment_path = os.path.join(options.default_experiment_directory, options.experiment_name)

    
    if not options.inner_dim:
        options.inner_dim = options.hidden_dim

    options.word2i_path = os.path.join(options.experiment_path, "word2i") if options.word2i_path is None else options.word2i_path
    options.char2i_path = os.path.join(options.experiment_path, "char2i")

    if options.length_to_size is not None:
        assert isinstance(options.length_to_size, str)
        parts = [x.split(':') for x in options.length_to_size.split(',')]
        options.length_to_size = {int(x[0]): int(x[1]) for x in parts}

    options.lowercase = not options.uppercase

    for k, v in options.__dict__.items():
        if type(v) == str and v.startswith('~'):
            options.__dict__[k] = os.path.expanduser(v)

    # Load model settings from a flags file.
    if options.model_flags is not None:
        flags_to_use = ('arch', 'compress', 'emb', 'hidden_dim', 'normalize', 'reconstruct_mode')
        options = init_with_flags_file(options, options.model_flags, flags_to_use)

    # Load any setting from a flags file.
    if options.flags is not None:
        options = init_with_flags_file(options, options.flags)

    return options


def configure(options : Args) -> Args:
    # Configure output paths for this experiment.
    configure_experiment(options.experiment_path, rank=options.local_rank)

    # Get logger.
    logger : logging.Logger = get_logger()

    # Print flags.
    logger.info(stringify_flags(options))
    save_flags(options, options.experiment_path)


if __name__ == '__main__':
    parser : Args = argument_parser()
    options : Args = parse_args(parser)
    configure(options)

    run(options)
