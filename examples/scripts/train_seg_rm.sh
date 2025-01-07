exp_prefix=$1
model_type=$2
dataset_type=$3
epoch=$4
segment_method=$5
entropy_threshold=$6
agg_func=$7

cache_dir="/data/denserlhf_training/reward_model/${exp_prefix}/${exp_prefix}_${model_type}_${dataset_type}_${epoch}epoch_segmethod-${segment_method}_entropy-${entropy_threshold}_aggfunc-${agg_func}"
data_dir="${cache_dir}/datasets"

export WANDB_API_KEY="YOUR_WANDB_API_KEY"
huggingface-cli login --token $HF_TOKEN

cd denserlhf
ACCELERATE_LOG_LEVEL=info

if [[ "$model_type" == *"phi3-instruct"* ]]; then
    model="microsoft/Phi-3-mini-4k-instruct"
    micro_train_batch_size=8
elif [[ "$model_type" == *"rlhflow_llama_3_sft_8b_v2"* ]]; then
    model="RLHFlow/LLaMA3-SFT-v2"
    micro_train_batch_size=4
elif [[ "$model_type" == *"meta_llama_3_1_instruct_8b"* ]]; then
    model="meta-llama/Llama-3.1-8B-Instruct"
    micro_train_batch_size=4
else
    model="EleutherAI/pythia-70m"
    micro_train_batch_size=4
fi

echo "reward model init from:" ${model}

torchrun --nproc_per_node=1 --nproc_per_node=8 cli/train_rm.py \
    --pretrain ${model} \
    --output_root_dir ${cache_dir}/outputs \
    --save_steps 256 \
    --logging_steps 16 \
    --eval_steps 512 \
    --micro_train_batch_size ${micro_train_batch_size} \
    --train_batch_size 128 \
    --max_epochs ${epoch} \
    --max_len 2048 \
    --max_prompt_len 1728 \
    --zero_stage 3 \
    --bf16 \
    --learning_rate 1e-6 \
    --flash_attn \
    --gradient_checkpointing \
    --dataset hendrydong/preference_700K \
    --exp_prefix ${exp_prefix} \
    --entropy_threshold ${entropy_threshold} \
    --agg_func ${agg_func} \
    --segment_method ${segment_method} \
    --load_checkpoint