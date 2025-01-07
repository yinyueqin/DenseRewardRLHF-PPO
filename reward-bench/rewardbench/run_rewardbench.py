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

# Run RewardBench (evaluate any reward model on any dataet)

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm
from transformers import AutoTokenizer

from rewardbench import (
    DPO_MODEL_CONFIG,
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
)


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
    parser.add_argument("--saved_path", type=str, default="saved", help="Path to save additional info.")
    args = parser.parse_args()

    ###############
    # Setup logging
    ###############
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
        quantized = config["quantized"]  # only Starling isn't quantized for now
        custom_dialogue = config["custom_dialogue"]
        pipeline_builder = config["pipeline_builder"]
        _ = config["model_type"]
        if custom_dialogue:
            raise NotImplementedError("Custom dialogue not implemented yet for simpler data formatting.")

    model_builder = config["model_builder"]
    ref_model_builder = ref_model_config["model_builder"]


    #########################
    # load dataset
    #########################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    
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

    dataset_type = "custom"
    
    print('\nmodel_type:', model_type)

    if args.dataset == "allenai/reward-bench":
        logger.info("Running core eval dataset.")
        from rewardbench import load_eval_dataset
        from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
        from rewardbench.utils import calculate_scores_per_section

        dataset, subsets = load_eval_dataset(
            core_set=True,
            conv=conv,
            custom_dialogue_formatting=False,
            tokenizer=tokenizer,
            logger=logger,
            keep_columns=["text_chosen", "text_rejected", "prompt", "chosen_text_only", "rejected_text_only"],
            model_type=model_type,
        )

        dataset_type = "rewardbench"

    if args.debug:
        dataset = dataset.select(range(10))

    print('dataset length:', len(dataset))


    logger.info("*** Load reward model ***")

    ############################
    # Load DPO model pipeline
    ############################
    if is_dpo:
        tokenizer.pad_token = tokenizer.eos_token
        # if no BOS token, set as pad token, e.g. QWEN models
        if tokenizer.bos_token is None:
            tokenizer.bos_token_id = tokenizer.eos_token_id
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }

        model = model_builder(
            args.model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs,
        )
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs,
        )

        # use internal inference functions in DPO trainer
        dpo = DPOInference(
            model,
            ref_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            # norm is norm, avg is average, sum is sum
        )

        # tokenize dataset
        column_names = list(dataset.features)

        tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=args.batch_size,
            collate_fn=DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=dpo.label_pad_token_id,
                is_encoder_decoder=dpo.is_encoder_decoder,
            ),
            # collate_fn = lambda x: x, # fix weird batching error
            shuffle=False,
            drop_last=False,
        )

    ############################
    # Load classifier model pipeline
    ############################
    else:
        reward_pipeline_kwargs = {
            "batch_size": args.batch_size,  # eval_args.inference_batch_size,
            "truncation": True,
            "padding": True,
            "max_length": args.max_length,
            "function_to_apply": "none",  # Compute raw logits
            "return_token_type_ids": False,
            "agg_func": args.agg_func,
            "num_thresholds": args.num_thresholds,
            'segment_method': args.segment_method,
            'model_type': model_type,
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
            "text-classification",  # often not used
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
        )
        
        if reward_pipe.tokenizer.pad_token_id is None:
            reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
            reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
        if reward_pipe.model.config.pad_token_id is None:
            reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

        # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
        if not check_tokenizer_chat_template(tokenizer):
            reward_pipe.tokenizer.add_eos_token = True

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        dataloader, model = accelerator.prepare(dataloader, reward_pipe.model)
        reward_pipe.model = model

    ############################
    # Run inference
    ############################
    
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

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        if is_dpo:
            rewards_chosen, rewards_rejected = dpo.inference_step(batch)
        else:
            entropy_thresholds = [args.entropy_threshold for _ in range(args.num_thresholds)]
            
            rewards_chosen, chosen_segment_num, chosen_sequence_length, chosen_segment_length, chosen_original_rewards, chosen_segment_position_rewards, chosen_additional_info = reward_pipe(
                batch["prompt"], batch["chosen_text_only"], entropy_thresholds, **reward_pipeline_kwargs
            )
            rewards_rejected, rejected_segment_num, rejected_sequence_length, rejected_segment_length, rejected_original_rewards, rejected_segment_position_rewards, rejected_additional_info = reward_pipe(
                batch["prompt"], batch["rejected_text_only"], entropy_thresholds, **reward_pipeline_kwargs
            )

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
            sample_info = {}
            for key in chosen_additional_info.keys():
                sample_info[key] = chosen_additional_info[key][i]
            chosen_all_additional_info.append(sample_info)
        
        for i in range(len(rejected_additional_info['decoded_sentences'])):
            sample_info = {}
            for key in rejected_additional_info.keys():
                sample_info[key] = rejected_additional_info[key][i]
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

    all_chosen_segment_num_tensor = torch.stack(all_chosen_segment_num)
    all_rejected_segment_num_tensor = torch.stack(all_rejected_segment_num)
    print('average chosen segment num:', torch.mean(all_chosen_segment_num_tensor.float()))
    print('average rejected segment num:', torch.mean(all_rejected_segment_num_tensor.float()))

    all_chosen_sequence_length_tensor = torch.stack(all_chosen_sequence_length)
    all_rejected_sequence_length_tensor = torch.stack(all_rejected_sequence_length)
    print('average chosen sequence length:', torch.mean(all_chosen_sequence_length_tensor.float()))
    print('average rejected sequence length:', torch.mean(all_rejected_sequence_length_tensor.float()))

    all_chosen_segment_length_tensor = torch.stack(all_chosen_segment_length)
    all_rejected_segment_length_tensor = torch.stack(all_rejected_segment_length)
    print('average chosen segment length:', torch.mean(all_chosen_segment_length_tensor.float()))
    print('average rejected segment length:', torch.mean(all_rejected_segment_length_tensor.float()))

    all_original_segment_rewards_tensor = torch.tensor(all_original_segment_rewards)
    original_reward_mean = torch.mean(all_original_segment_rewards_tensor)
    original_reward_std = torch.std(all_original_segment_rewards_tensor).clamp(min=1e-8)
    print(f"Original Segment Rewards - Mean: {original_reward_mean}, Std: {original_reward_std}")

    all_final_segment_rewards = torch.tensor(all_final_segment_rewards)
    final_reward_mean = torch.mean(all_final_segment_rewards)
    final_reward_std = torch.std(all_final_segment_rewards).clamp(min=1e-8)
    print(f"Final Segment Rewards - Mean: {final_reward_mean}, Std: {final_reward_std}")

    accuracy = sum(results) / len(results)
    logger.info(f"Results: {accuracy}, on {len(results)} prompts")

    processed_rewards = defaultdict(list)
    for item in all_segment_position_rewards:
        for position, reward in item.items():
            if isinstance(reward, (int, float)):
                processed_rewards[position].append(reward)
            elif isinstance(reward, (list, np.ndarray)):
                processed_rewards[position].append(np.mean(reward))
            else:
                print(f"Unexpected reward type: {type(reward)} for position {position}")

    mean_rewards = {pos: np.mean(rewards) for pos, rewards in processed_rewards.items()}
    std_rewards = {pos: np.std(rewards) for pos, rewards in processed_rewards.items()}

    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)

    with open(os.path.join(saved_path, f'{model_type}_{dataset_type}_{args.entropy_threshold}_{args.agg_func}_{args.segment_method}_segment_position_rewards_samples{len(dataset)}.json'), 'w') as f:
        json.dump({
            'mean': {k: float(v) for k, v in mean_rewards.items()},
            'std': {k: float(v) for k, v in std_rewards.items()}
        }, f)

    chosen_filename = os.path.join(saved_path, f'{model_type}_{dataset_type}_{args.entropy_threshold}_{args.agg_func}_{args.segment_method}_chosen_segment_text_rewards.json')
    save_data_to_json(chosen_all_additional_info, chosen_filename)

    rejected_filename = os.path.join(saved_path, f'{model_type}_{dataset_type}_{args.entropy_threshold}_{args.agg_func}_{args.segment_method}_rejected_segment_text_rewards.json')
    save_data_to_json(rejected_all_additional_info, rejected_filename)


    if args.dataset == "allenai/reward-bench":
        out_dataset = dataset.add_column("results", results)
        if args.debug:
            subsets = subsets[:10]
        out_dataset = out_dataset.add_column("subsets", subsets)
        out_dataset = out_dataset.to_pandas()  # I know this is meh

        results_grouped = {}
        present_subsets = np.unique(out_dataset["subsets"])
        for subset in present_subsets:
            subset_dataset = out_dataset[out_dataset["subsets"] == subset]
            num_correct = sum(subset_dataset["results"])
            num_total = len(subset_dataset["results"])
            logger.info(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
            results_grouped[subset] = num_correct / num_total

        results_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        logger.info(f"Results: {results_section}")

    
    ############################
    # compile scores
    ############################
    # save score in json to args.output_dir + args.model + ".json"
    output_path = args.output_dir + args.model + ".json"
    dirname = os.path.dirname(output_path)
    os.makedirs(dirname, exist_ok=True)

    # remove old data
    if os.path.exists(output_path):
        os.remove(output_path)

    with open(output_path, "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "num_prompts": len(results),
                "model": args.model,
                "ref_model": args.ref_model,
                "tokenizer": tokenizer_path,
                "chat_template": args.chat_template,
                "extra_results": results_grouped if args.dataset == "allenai/reward-bench" else None,
            },
            f,
        )

if __name__ == "__main__":
    main()