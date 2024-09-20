if [ "$#" -ne 2 ]
then
    echo "illegal number of parameters"
    echo Usage:
    echo "sh run_several.sh <experiment_name> <num_runs>"
    exit 1
fi

name=$1
runs=$2
i=1
export PYTHONPATH=$(pwd)/pytorch:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

while [ $i -lt $((runs+1)) ]
do
    echo "running experiment $i"

    python -m torch.distributed.launch --nproc_per_node=4 pytorch/diora/scripts/train.py \
        --experiment_path ./log/${name}_${i} \
        --arch mlp-shared \
        --batch_size 32 \
        --data_type flat \
        --validattion_data_type flat \
        --elmo_cache_dir ./cache \
        --emb elmo \
        --hidden_dim 400 \
        --k_neg 100 \
        --log_every_batch 100 \
        --lr 2e-3 \
        --normalize unit \
        --reconstruct_mode softmax \
        --save_after 1000 \
        --train_filter_length 20 \
        --train_path ./data/ptb-dev-wr \ #~/babylm_data/babylm_100M/qed.train \
        --validation_path ./data/ptb/ptb-dev-wr \
        --max_step 300000 \
        --cuda
        
    true $((i=i+1))
done

echo "done"
