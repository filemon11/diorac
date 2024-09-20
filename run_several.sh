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
#export TOKENIZERS_PARALLELISM=false
export NGPUS=8
export NNODES=1
export CUDA_LAUNCH_BLOCKING=1

while [ $i -lt $((runs+1)) ]
do
    echo "running experiment $i"

# max 32 per A100 GPU
#python pytorch/diora/scripts/train.py \
torchrun --nnodes=$NNODES --nproc_per_node=$NGPUS --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 pytorch/diora/scripts/train.py \
    --experiment_path ./log/${name}_${i} \
    --arch treelstm \
    --batch_size 64 \
    --train_data_type flat \
    --elmo_cache_dir ./cache \
    --emb elmo \
    --hidden_dim 292 \
    --inner_dim 2371 \
    --k_neg 100 \
    --log_every_batch 100 \
    --lr 0.0115296 \
    --normalize unit \
    --reconstruct_mode margin \
    --save_after 1000 \
    --train_filter_length 40 \
    --train_path ./data/babylm/processed_100M/all.txt \
    --max_step 1000000000 \
    --max_epoch 1000000000 \
    --num_workers 0 \
    --cuda \
    --accum_iter 1 \
    --local_rank 0 \
    --multigpu
        
    true $((i=i+1))
done

echo "done"

#~/babylm_data/babylm_100M/all.train \
