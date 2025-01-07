exp_prefix=$1
model_type=$2
dataset_type=$3
num_episodes=$4
segment_method=$5
entropy_threshold=$6
agg_func=$7
ppo_reward_type=$8
value_clip=$9
reward_fit_dataset=${10}

export WANDB_API_KEY="YOUR_WANDB_API_KEY"
huggingface-cli login --token $HF_TOKEN

reward_mean=0
reward_std=1

if [ "$model_type" = "phi3-instruct" ]; then
    pretrain_model="microsoft/Phi-3-mini-4k-instruct"
    micro_train_batch_size=4
    micro_rollout_batch_size=16

    if [ "$entropy_threshold" = "1.75" ]; then
        rm_model_dir="yyqoni/Phi-3-mini-4k-instruct-segment-rm-700k"
    elif [ "$entropy_threshold" = "0" ]; then
        rm_model_dir="yyqoni/Phi-3-mini-4k-instruct-token-rm-700k"
    elif [ "$entropy_threshold" = "1000" ]; then
        reward_mean=-0.51953125
        reward_std=2.15625
        rm_model_dir="yyqoni/Phi-3-mini-4k-instruct-bandit-rm-700k"
    fi
elif [ "$model_type" = "rlhflow_llama_3_sft_8b_v2" ]; then
    pretrain_model="RLHFlow/LLaMA3-SFT-v2"
    micro_train_batch_size=2
    micro_rollout_batch_size=4
    if [ "$entropy_threshold" = "2" ]; then
        rm_model_dir="yyqoni/rlhflow-llama-3-sft-8b-v2-segment-rm-700k"
    elif [ "$entropy_threshold" = "0" ]; then
        rm_model_dir="yyqoni/rlhflow-llama-3-sft-8b-v2-token-rm-700k"
    elif [ "$entropy_threshold" = "1000" ]; then
        reward_mean=-0.81640625
        reward_std=2.953125
        rm_model_dir="yyqoni/rlhflow-llama-3-sft-8b-v2-bandit-rm-700k"
    fi
elif [ "$model_type" = "meta_llama_3_1_instruct_8b" ]; then
    pretrain_model="meta-llama/Llama-3.1-8B-Instruct"
    micro_train_batch_size=2
    micro_rollout_batch_size=4
    if [ "$entropy_threshold" = "2" ]; then
        rm_model_dir="yyqoni/meta-llama-3.1-instruct-8b-segment-rm-700k"
    elif [ "$entropy_threshold" = "0" ]; then
        rm_model_dir="yyqoni/meta-llama-3-1-instruct-8b-token-rm-700k"
    elif [ "$entropy_threshold" = "1000" ]; then
        reward_mean=-0.828125
        reward_std=2.9375
        rm_model_dir="yyqoni/meta-llama-3-1-instruct-8b-bandit-rm-700k"
    fi
else
    pretrain_model=EleutherAI/pythia-70m
    rm_model_dir=EleutherAI/pythia-70m
fi


policy_model_dir="/data/openrlhf_training/ppo_policy/${exp_prefix}/${exp_prefix}_${model_type}_${dataset_type}_Episodes${num_episodes}_seg_mtd${segment_method}_Entropy${entropy_threshold}_agg_func${agg_func}_PPO${ppo_reward_type}_value_clip${value_clip}_reward_reward_fit_dataset_${reward_fit_dataset}/"

echo "Pretrain Model: $pretrain_model"
echo "Reward Model Directory: $rm_model_dir"
echo "Policy Model Directory: $policy_model_dir"
echo "Reward Mean: $reward_mean"
echo "Reward Std: $reward_std"

cd denserlhf

torchrun --nproc_per_node=1 --nproc_per_node=8 cli/train_ppo.py \
    --pretrain ${pretrain_model} \
    --reward_pretrain ${rm_model_dir} \
    --output_root_dir ${policy_model_dir}/outputs \
    --exp_prefix ${exp_prefix} \
    --save_steps 5 \
    --logging_steps 1 \
    --micro_train_batch_size ${micro_train_batch_size} \
    --train_batch_size 128 \
    --micro_rollout_batch_size ${micro_rollout_batch_size} \
    --rollout_batch_size 1024 \
    --num_episodes ${num_episodes} \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --prompt_data argilla/ultrafeedback-binarized-preferences-cleaned \
    --max_samples 800000 \
    --actor_init_on_gpu \
    --flash_attn \
    --gradient_checkpointing \
    --entropy_threshold ${entropy_threshold} \
    --ppo_reward_type ${ppo_reward_type} \
    --reward_mean ${reward_mean} \
    --reward_std ${reward_std} \
    --value_clip ${value_clip} \
    --normalize_reward \
    --segment_method ${segment_method} \
    --agg_func ${agg_func} \
    --reward_fit_dataset ${reward_fit_dataset} \
    --load_checkpoint \
    --adam_offload