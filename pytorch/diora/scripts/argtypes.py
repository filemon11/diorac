from dataclasses import dataclass

import argparse

from typing import Optional, Dict, Literal

@dataclass()
class Args(argparse.Namespace):
    experiment_name : str
    experiment_path : str
    load_model_path : str
    embeddings_path : str
    w2i_from_tokenizer : bool
    all_from_w2i : bool
    synthetic_nexamples : bool
    synthetic_vocabsize : int
    synthetic_embeddingsize : int
    synthetic_minlen : int
    synthetic_maxlen : int
    synthetic_seed : int
    synthetic_length : int
    use_synthetic_embeddings : bool
    arch : str
    hidden_dim : int
    inner_dim : int | None
    normalize : bool
    compress : bool
    emb : int
    margin : int
    elmo_options_path : str
    elmo_weights_path : str
    elmo_cache_dir : str
    train_dataset_size : int
    validation_dataset_size : int
    max_epoch : int
    max_step : int
    finetune : bool
    finetune_after : int
    postprocess : bool
    retain_file_order : bool
    lr : float
    default_experiment_directory : str
    data_type : str
    train_filter_length : int
    validation_filter_length : int
    train_path : str
    validation_path : str
    batch_size : int
    word2i_path : str
    uppercase : bool
    local_rank : int
    tokenizer : str
    k_neg : int
    freq_dist_power : int
    reconstruct_mode : Literal['margin', 'softmax']
    cuda : bool
    multigpu : bool
    num_workers : int
    seed : int
    train_data_type : Optional[str]
    validation_data_type : Optional[str]
    transformed_validation_trees_path : Optional[str]
    validation_batch_size : Optional[int]
    length_to_size : str | Dict[int, int] | None
    git_branch_name : Optional[str]
    git_sha : Optional[str]
    git_dirty : Optional[str]
    master_addr : str
    master_port : str
    world_size : int
    uuid : Optional[str]
    model_flags : str
    flags : str
    log_every_batch : int
    save_latest : int
    save_distinct : int 
    save_after : int
    save_init : int
    fragments : Literal["random", "all", False]
    add_future : bool
    early_stopping : int
    accum_iter : int
    use_validation : bool
    hyperopt : bool
    n_trials : int
