# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run RewardBench (evaluate any reward model on any dataset)

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm
from transformers import AutoTokenizer
import datasets
from collections import defaultdict

from rewardbench import (
    DPO_MODEL_CONFIG,
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
)

from torch.utils.data import Dataset
from accelerate.utils import gather_object

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [tensor_to_serializable(v) for v in obj]
    else:
        return str(obj)

def format_for_json(data):
    """Format the data to be more readable in JSON"""
    if isinstance(data, dict):
        return {k: format_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], (list, dict)):
            return [format_for_json(item) for item in data]
        else:
            formatted = ", ".join(map(str, data))
            return formatted if len(formatted) < 50 else data
    else:
        return data

def save_data_to_json(data, filename):
    serializable_data = tensor_to_serializable(data)
    
    formatted_data = format_for_json(serializable_data)
    
    with open(filename, 'w') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

class RewardDataset(Dataset):
    def __init__(self, data, tokenizer, model_type):
        self.data = data
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = item["chosen"][:-1]
        chosen_response = item["chosen"][-1]
        rejected_response = item["rejected"][-1]

        chosen_text = self.tokenizer.apply_chat_template([chosen_response], tokenize=False, add_generation_prompt=False)
        rejected_text = self.tokenizer.apply_chat_template([rejected_response], tokenize=False, add_generation_prompt=False)
        prompt_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)

        if 'phi' in self.model_type.lower():
            if chosen_text.endswith(self.tokenizer.eos_token):
                chosen_text = chosen_text[:-len(self.tokenizer.eos_token)].strip()
                chosen_text = chosen_text.replace("<|assistant|>\n", "")
            if rejected_text.endswith(self.tokenizer.eos_token):
                rejected_text = rejected_text[:-len(self.tokenizer.eos_token)].strip()
                rejected_text = rejected_text.replace("<|assistant|>\n", "")
            if prompt_text.endswith(self.tokenizer.eos_token):
                prompt_text = prompt_text[:-len(self.tokenizer.eos_token)]
                prompt_text = prompt_text + "<|assistant|>\n"
                
        elif self.model_type == "rlhflow_llama_3_sft_8b_v2":
            chosen_text = chosen_text.replace("<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
            rejected_text = rejected_text.replace("<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
            prompt_text = prompt_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"

        elif self.model_type == 'meta_llama_3_1_instruct_8b':
            marker = '<|eot_id|>'
            pos = chosen_text.find('<|eot_id|>')
            chosen_text = chosen_text[pos + len(marker):].replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
            rejected_text = rejected_text[pos + len(marker):].replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
            prompt_text = prompt_text[pos + len(marker):] + "<|start_header_id|>assistant<|end_header_id|>\n\n"

        
        if idx < 10:
            print('\n after apply_chat_template prompt:', prompt_text, 'idx:', idx)
            print('\n after apply_chat_template chosen_response:', chosen_text, 'idx:', idx)
            print('\n after apply_chat_template rejected_response:', rejected_text, 'idx:', idx)
        
        return {
            "prompt": prompt_text,
            "chosen_text_only": chosen_text,
            "rejected_text_only": rejected_text,
        }

def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    chosen_responses = [item["chosen_text_only"] for item in batch]
    rejected_responses = [item["rejected_text_only"] for item in batch]
    return {
        "prompt": prompts,
        "chosen_text_only": chosen_responses,
        "rejected_text_only": rejected_responses
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a reward model.")

    # core args
    parser.add_argument("--dataset", type=str, default="allenai/reward-bench", help="The dataset to evaluate on.")
    parser.add_argument("--split", type=str, default=None, help="The split to evaluate on.")
    parser.add_argument("--model", type=str, required=True, help="The model to evaluate.")
    parser.add_argument("--model_id", type=str, default=None, help="The model id to evaluate.")
    parser.add_argument("--ref_model", type=str, default=None, help="The reference model to compare against.")
    parser.add_argument("--tokenizer", type=str, default=None, help="The tokenizer to use (defaults to model).")
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="The chat template to use (defaults to from tokenizer, from chattemplate).",
    )

    # inference args
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size to use.")
    parser.add_argument("--max_length", type=int, default=512, help="The max length to use.")

    # system args
    parser.add_argument("--load_json", action="store_true", default=False, help="Load dataset as json.")
    parser.add_argument("--trust_remote_code", action="store_true", default=False, help="Trust remote code.")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode.")
    parser.add_argument("--output_dir", type=str, default="results", help="The output directory to save results.")
    parser.add_argument("--save_all", action="store_true", default=False, help="Save all results.")
    parser.add_argument("--entropy_threshold", type=float, default=0.0, help="entropy threshold.")
    parser.add_argument("--num_thresholds", type=int, default=1, help="Number of thresholds.")
    parser.add_argument("--agg_func", type=str, default="avg", help="Aggregation function.")
    parser.add_argument("--segment_method", type=str, default="peak", help="Segmentation method.")
    parser.add_argument("--chosen_samples_num", type=int, default=100, help="Number of chosen samples.")
    args = parser.parse_args()

    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"""
        Running reward model on {args.model} with chat template {args.chat_template}
        Using trust remote code: {args.trust_remote_code}
            """.strip())

    is_dpo = False
    MODEL_CONFIGS = REWARD_MODEL_CONFIG
    REF_MODEL_CONFIGS = DPO_MODEL_CONFIG

    if args.chat_template:
        from fastchat.conversation import get_conv_template

        conv = get_conv_template(args.chat_template)
    else:
        conv = None

    if args.model_id in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.model_id]
    else:
        config = MODEL_CONFIGS["default"]
    
    ref_model_config = REF_MODEL_CONFIGS["default"]
    logger.info(f"""
        Using reward model config: {config}
        Using reference model config: {ref_model_config}
            """.strip())

    if not is_dpo:
        quantized = config["quantized"]
        custom_dialogue = config["custom_dialogue"]
        pipeline_builder = config["pipeline_builder"]
        _ = config["model_type"]
        if custom_dialogue:
            raise NotImplementedError("Custom dialogue not implemented yet for simpler data formatting.")

    model_builder = config["model_builder"]
    ref_model_builder = ref_model_config["model_builder"]

    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)

    logger.info("*** Load reward model ***")
    if 'phi3-instruct' in args.model.lower():
        model_type = 'phi3-instruct'
    elif 'phi' in args.model.lower():
        model_type = 'phi_sft'
    elif 'rlhflow_llama_3_sft_8b_v2' in args.model.lower():
        model_type = 'rlhflow_llama_3_sft_8b_v2'
    elif 'meta_llama_3_1_instruct_8b' in args.model.lower():
        model_type = 'meta_llama_3_1_instruct_8b'
    else:
        model_type = 'debug'
        
    print('\nmodel_type:', model_type)
    
    reward_pipeline_kwargs = {
        "batch_size": args.batch_size,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",
        "return_token_type_ids": False,
        "agg_func": args.agg_func,
        "num_thresholds": args.num_thresholds,
        'segment_method': args.segment_method,
        'model_type': model_type
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {"device_map": {"": current_device}}
    
    if config.get("external_model_kwargs"):
        model_kwargs.update(config["external_model_kwargs"])

    model = model_builder(args.model, **model_kwargs)
        
    ref_model_kwargs = {
        "device_map": {"": current_device},
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }

    ref_model = ref_model_builder(
        args.ref_model,
        **ref_model_kwargs,
    )

    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id
    
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    dataset = datasets.load_dataset('hendrydong/preference_700K', split='train')

    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(args.chosen_samples_num))

    dataset = RewardDataset(dataset, reward_pipe.tokenizer, model_type)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn, 
        drop_last=False
    )

    dataloader = accelerator.prepare(dataloader)
    reward_pipe.model = model
    accelerator.wait_for_everyone()

    results = []
    scores_chosen = []
    scores_rejected = []
    all_chosen_segment_num = []
    all_rejected_segment_num = []
    all_chosen_sequence_length = []
    all_rejected_sequence_length = []
    all_chosen_segment_length = []
    all_rejected_segment_length = []
    all_original_segment_rewards = []
    all_final_segment_rewards = []
    all_segment_position_rewards = []
    chosen_all_additional_info = []
    rejected_all_additional_info = []

    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    
    for step, batch in enumerate(dataloader):
        progress_bar.update(1)
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        if not is_dpo:
            entropy_thresholds = [args.entropy_threshold] * args.num_thresholds
            rewards_chosen, chosen_segment_num, chosen_sequence_length, chosen_segment_length, chosen_original_rewards, chosen_segment_position_rewards, chosen_additional_info = reward_pipe(
                batch["prompt"], batch["chosen_text_only"], entropy_thresholds, **reward_pipeline_kwargs
            )
            rewards_rejected, rejected_segment_num, rejected_sequence_length, rejected_segment_length, rejected_original_rewards, rejected_segment_position_rewards, rejected_additional_info = reward_pipe(
                batch["prompt"], batch["rejected_text_only"], entropy_thresholds, **reward_pipeline_kwargs
            )

            if step % 100 == 0 and accelerator.is_local_main_process:
                print('\nrewards_chosen:', rewards_chosen)
                print('\nrewards_rejected:', rewards_rejected)

            all_chosen_segment_num.extend(chosen_segment_num)
            all_rejected_segment_num.extend(rejected_segment_num)
            all_chosen_sequence_length.extend(chosen_sequence_length)
            all_rejected_sequence_length.extend(rejected_sequence_length)
            all_chosen_segment_length.extend(chosen_segment_length)
            all_rejected_segment_length.extend(rejected_segment_length)
            all_original_segment_rewards.extend(chosen_original_rewards)
            all_original_segment_rewards.extend(rejected_original_rewards)
            all_final_segment_rewards.extend(rewards_chosen)
            all_final_segment_rewards.extend(rewards_rejected)
            all_segment_position_rewards.extend(chosen_segment_position_rewards)
            all_segment_position_rewards.extend(rejected_segment_position_rewards)

            for i in range(len(chosen_additional_info['decoded_sentences'])):
                sample_info = {key: chosen_additional_info[key][i] for key in chosen_additional_info.keys()}
                chosen_all_additional_info.append(sample_info)
            
            for i in range(len(rejected_additional_info['decoded_sentences'])):
                sample_info = {key: rejected_additional_info[key][i] for key in rejected_additional_info.keys()}
                rejected_all_additional_info.append(sample_info)

            if isinstance(rewards_chosen[0], dict):
                score_chosen_batch = [result["score"] for result in rewards_chosen]
                score_rejected_batch = [result["score"] for result in rewards_rejected]
            else:
                score_chosen_batch = rewards_chosen.float().cpu().numpy().tolist()
                score_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()

            for chosen_scores, rejected_scores in zip(score_chosen_batch, score_rejected_batch):
                chosen_best, rejected_best = max(
                    zip(chosen_scores, rejected_scores), key=lambda x: x[0] - x[1]
                )
                scores_chosen.append(chosen_best)
                scores_rejected.append(rejected_best)
                results.append(1 if chosen_best > rejected_best else 0)

    def calculate_mean(data):
        tensor_data = torch.tensor(data, device='cpu')
        mean = torch.mean(tensor_data.float())
        return mean.item()
    
    def calculate_std(data):
        tensor_data = torch.tensor(data, device='cpu')
        std = torch.std(tensor_data.float())
        return std.item()

    local_results = {
        'chosen_segment_num_mean': calculate_mean(all_chosen_segment_num),
        'rejected_segment_num_mean': calculate_mean(all_rejected_segment_num),
        'chosen_sequence_length_mean': calculate_mean(all_chosen_sequence_length),
        'rejected_sequence_length_mean': calculate_mean(all_rejected_sequence_length),
        'chosen_segment_length_mean': calculate_mean(all_chosen_segment_length),
        'rejected_segment_length_mean': calculate_mean(all_rejected_segment_length),
        'original_segment_rewards_mean': calculate_mean(all_original_segment_rewards),
        'final_segment_rewards_mean': calculate_mean(all_final_segment_rewards),
        'final_segment_rewards_std': calculate_std(all_final_segment_rewards),
    }
    

    print(f"Local rank: {accelerator.local_process_index}, local_results: {local_results}")

    def move_to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: move_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_cpu(v) for v in obj]
        else:
            return obj
        
    all_segment_position_rewards = move_to_cpu(all_segment_position_rewards)
    gathered_segment_position_rewards = gather_object(all_segment_position_rewards)

    
    torch.cuda.empty_cache()
   
    accelerator.wait_for_everyone()

    gathered_local_results = gather_object([local_results])

    if accelerator.is_main_process:
        final_results = defaultdict(float)
        num_processes = accelerator.num_processes

        for i, local_result in enumerate(gathered_local_results):
            print(f"Type of local_result {i}: {type(local_result)}")
            print(f"Content of local_result {i}: {local_result}")
            
            if isinstance(local_result, dict):
                for key, value in local_result.items():
                    final_results[key] += value / num_processes
            else:
                print(f"Warning: local_result {i} is not a dictionary")

        stats = {}
        stats['chosen_segment_num'] = {'mean': final_results['chosen_segment_num_mean']}
        stats['rejected_segment_num'] = {'mean': final_results['rejected_segment_num_mean']}
        stats['chosen_sequence_length'] = {'mean': final_results['chosen_sequence_length_mean']}
        stats['rejected_sequence_length'] = {'mean': final_results['rejected_sequence_length_mean']}
        stats['chosen_segment_length'] = {'mean': final_results['chosen_segment_length_mean']}
        stats['rejected_segment_length'] = {'mean': final_results['rejected_segment_length_mean']}
        stats['original_segment_rewards'] = {'mean': final_results['original_segment_rewards_mean']}
        stats['final_segment_rewards'] = {'mean': final_results['final_segment_rewards_mean'], 'std': final_results['final_segment_rewards_std']}

        print('stats:', stats)

        processed_rewards = defaultdict(list)
        for item in gathered_segment_position_rewards:
            for position, reward in item.items():
                if isinstance(reward, (int, float)):
                    processed_rewards[position].append(reward)
                elif isinstance(reward, (list, np.ndarray)):
                    processed_rewards[position].append(np.mean(reward))

        print('num of processed_rewards:', len(processed_rewards))
        mean_rewards = {pos: np.mean(rewards) for pos, rewards in processed_rewards.items()}
        std_rewards = {pos: np.std(rewards) for pos, rewards in processed_rewards.items()}

        accuracy = sum(results) / len(results)
        logger.info(f"Results: {accuracy:.4f}, on {len(results)} prompts")

        dataset_type = 'preference700k'

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'{model_type}_{dataset_type}_{args.entropy_threshold}_{args.agg_func}_{args.segment_method}_segment_position_rewards_samples{len(dataset)}.json'), 'w') as f:
            json.dump({
                'mean': {k: float(v) for k, v in mean_rewards.items()},
                'std': {k: float(v) for k, v in std_rewards.items()}
            }, f)
        chosen_filename = os.path.join(args.output_dir, f'{model_type}_{dataset_type}_{args.entropy_threshold}_{args.agg_func}_{args.segment_method}_chosen_segment_text_rewards_samples{len(dataset)}.json')
        save_data_to_json(chosen_all_additional_info, chosen_filename)
        rejected_filename = os.path.join(args.output_dir, f'{model_type}_{dataset_type}_{args.entropy_threshold}_{args.agg_func}_{args.segment_method}_rejected_segment_text_rewards_samples{len(dataset)}.json')
        save_data_to_json(rejected_all_additional_info, rejected_filename)
        stats_filename = os.path.join(args.output_dir, f'{model_type}_{dataset_type}_{args.entropy_threshold}_{args.agg_func}_{args.segment_method}_evaluation_statistics_samples{len(dataset)}.json')
        save_data_to_json(stats, stats_filename)

if __name__ == "__main__":
    main()
