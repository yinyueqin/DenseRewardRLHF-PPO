model=${1}
segment_method=${2}
entropy_threshold=${3}
agg_func=${4}

echo "model: ${model}"
echo "segment_method: ${segment_method}"
echo "entropy_threshold: ${entropy_threshold}"
echo "agg_func: ${agg_func}"
    
cd reward-bench

huggingface-cli login --token $HF_TOKEN

batch_size=8

if [[ "${model}" == *"Phi"* ]]; then
    ref_model_id="microsoft/Phi-3-mini-4k-instruct"
elif [[ "${model}" == *"llama"* && "${model}" == *"sft"* ]]; then
    ref_model_id="RLHFlow/LLaMA3-SFT-v2"
elif [[ "${model}" == *"llama"* && "${model}" == *"meta"* ]]; then
    ref_model_id="meta-llama/Llama-3.1-8B-Instruct"
fi

echo "model id: ${model}"
echo "ref model id: ${ref_model_id}"

echo "Run calc_train_reward_mean_std"

accelerate launch rewardbench/calc_train_reward_mean_std.py \
    --model="${model}" \
    --model_id="denserlhf_seg_rm" \
    --save_all \
    --trust_remote_code \
    --ref_model="${ref_model_id}" \
    --entropy_threshold="${entropy_threshold}" \
    --agg_func="${agg_func}" \
    --segment_method="${segment_method}" \
    --batch_size=${batch_size} \
    --chosen_samples_num=60000 \
    --output_dir="/data/saved_results/preference700k/"
