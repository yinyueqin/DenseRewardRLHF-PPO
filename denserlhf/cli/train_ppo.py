import argparse
import itertools
import math
import os
from datetime import datetime

import torch
from transformers.trainer import get_scheduler

from denserlhf.datasets import CustomPromptDataset
from denserlhf.models import Actor, get_llm_for_sequence_regression
from denserlhf.trainer import PPOTrainer
from denserlhf.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()
    if strategy.is_rank_0():
        os.makedirs(args.output_root_dir, exist_ok=True)
    args.save_path = os.path.join(args.output_root_dir, "models")
    args.ckpt_path = os.path.join(args.output_root_dir, "ckpts")
    strategy.print("Output result to:", args.save_path, args.ckpt_path)

    strategy.print("\nargs.entropy_threshold:", args.entropy_threshold)
    strategy.print("\nargs.ppo_reward_type:", args.ppo_reward_type)
    strategy.print("\nargs.agg_func:", args.agg_func)
    strategy.print("\nargs.segment_method:", args.segment_method)
    strategy.print("\nargs.model_type:", args.model_type)
    strategy.print("\nargs.reward_fit_dataset:", args.reward_fit_dataset)
    strategy.print("\nargs.init_kl_coef:", args.init_kl_coef)
    strategy.print("\nargs.gamma:", args.gamma)
    strategy.print("\nargs.num_episodes:", args.num_episodes)
    strategy.print("\nargs.actor_learning_rate:", args.actor_learning_rate)
    strategy.print("\nargs.critic_learning_rate:", args.critic_learning_rate)

    # configure model
    # load huggingface model
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        strategy=strategy,
    )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    if args.init_critic_from_rm:
        strategy.print('\ninit critic from reward model')
        args.critic_pretrain = args.reward_pretrain
        critic = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "critic",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=args.value_head_prefix,
            init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
        )
        strategy.print('init_value_head: ', strategy.args.pretrain == strategy.args.critic_pretrain)
    else:
        strategy.print('\ninit critic from sft')
        args.critic_pretrain = args.pretrain
        critic = get_llm_for_sequence_regression(
            args.critic_pretrain,
            "critic",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=args.value_head_prefix,
            init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
            )
        strategy.print('init_value_head: ', strategy.args.pretrain == strategy.args.critic_pretrain)

    if not args.remote_rm_url:
        reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=args.value_head_prefix,
            reward_mean=args.reward_mean,
            reward_std=args.reward_std,
        )
    else:
        reward_model = None

    strategy.print("reward normalization status: {}".format(args.normalize_reward))
    strategy.print("mean: {}, std {}".format(critic.mean, critic.std))


    strategy.print(actor)
    strategy.print(critic)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # load weights for reference actor
    initial_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=False),
    )

    if args.enable_ema:
        ema_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=True),
        )
    else:
        ema_model = None

    # gradient_checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        critic.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )
    critic_optim = strategy.create_optimizer(
        critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )

    # prepare datasets
    prompts_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        return_eval=False,
        train_split=args.prompt_split,
    )
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = CustomPromptDataset(prompts_data, tokenizer, strategy, model_type=args.model_type)

    print('len(prompts_dataset):', len(prompts_dataset))

    if args.pretrain_data:
        pretrain_data = blending_datasets(
            args.pretrain_data,
            args.pretrain_data_probs,
            strategy,
            args.seed,
            return_eval=False,
            train_split=args.pretrain_split,
        )
        pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        pretrain_dataset = SFTDataset(
            pretrain_data.select(
                range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset) * args.n_samples_per_prompt))
            ),
            tokenizer,
            pretrain_max_len,
            strategy,
            pretrain_mode=True,
        )

    # prepare dataloader
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True, collate_fn=lambda x: x)

    if args.pretrain_data:
        pretrain_dataloader = itertools.cycle(
            iter(
                strategy.setup_dataloader(
                    pretrain_dataset,
                    args.micro_train_batch_size,
                    True,
                    True,
                    pretrain_dataset.collate_fn,
                )
            )
        )
    else:
        pretrain_dataloader = None

    # configure scheduler
    num_update_steps_per_episodes = len(prompts_dataset) // args.train_batch_size * args.max_epochs
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    actor_scheduler = get_scheduler(
        "cosine_with_min_lr",
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
    )

    critic_scheduler = get_scheduler(
        "cosine_with_min_lr",
        critic_optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
    )
    
    strategy.print("scheduler created")
    # prepare models/optimizers...
    (
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
    ) = strategy.prepare(
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
        is_rlhf=True,
    )

    strategy.print("model prepared")

    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True)

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        print("\ndetected existing checkpoint, resume from", args.ckpt_path)
        actor_path = os.path.join(args.ckpt_path, "_actor")
        actor_load_path, states = strategy.load_ckpt(actor.model, actor_path)
        assert actor_load_path is not None, f"Failed to load checkpoint from {actor_path}"
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loading actor's checkpoint from {actor_path}, consumed_samples: {consumed_samples}")

        critic_path = os.path.join(args.ckpt_path, "_critic")
        strategy.print(f"Loading critic's checkpoint from {critic_path}")
        critic_load_path, _ = strategy.load_ckpt(critic, critic_path)
        assert critic_load_path is not None, f"Failed to load checkpoint from {critic_path}"
    else:
        if strategy.is_rank_0():
            print("\nno existing checkpoint found, training from scratch")
    


    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = PPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        ema_model,
        actor_optim,
        critic_optim,
        actor_scheduler,
        critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ema_beta=0.992,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        # fro GPT generation
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # remote reward model
        remote_rm_url=args.remote_rm_url,
        entropy_threshold=args.entropy_threshold,
        num_thresholds=args.num_thresholds,
        ppo_reward_type=args.ppo_reward_type,
        agg_func=args.agg_func,
        segment_method=args.segment_method,
        model_type=args.model_type,
        reward_fit_dataset=args.reward_fit_dataset,
    )


    trainer.fit(args, prompts_dataloader, pretrain_dataloader, consumed_samples, num_update_steps_per_episodes)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(
        ema_model if args.enable_ema else actor,
        tokenizer,
        args.save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--output_root_dir", type=str)


    # PPO
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=5e-7)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="chosen", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="denserlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # Other parameters for segment reward model
    parser.add_argument("--exp_prefix", type=str, default="debug")
    parser.add_argument("--entropy_threshold", type=float, default=2)
    parser.add_argument("--num_thresholds", type=int, default=1)
    parser.add_argument("--ppo_reward_type", type=str, default="segment_normalization")
    parser.add_argument("--agg_func", type=str, default="avg")
    parser.add_argument("--segment_method", type=str, default="peak_v2")
    parser.add_argument("--reward_fit_dataset", type=str, default="ultrafeedback")

    parser.add_argument("--reward_mean", type=float, default=100)
    parser.add_argument("--reward_std", type=float, default=-100)
    parser.add_argument("--norm_params_path", type=str, default='param.json')

    parser.add_argument("--init_critic_from_rm", action="store_true", default=False, help="if True, init critic from reward model")

    args = parser.parse_args()

    if 'debug' in args.exp_prefix:
        args.wandb_project = 'debug'
    else:
        args.wandb_project = 'ppo_training_end_compare'

    args.use_wandb = True

    if args.critic_pretrain is None:
        if not args.remote_rm_url:
            args.critic_pretrain = args.reward_pretrain
        else:
            args.critic_pretrain = args.pretrain

    if args.input_template and not "{}" in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if 'phi' in args.pretrain.lower() and 'instruct' in args.pretrain.lower():
        model_type = 'phi3-instruct'
    elif 'phi' in args.pretrain.lower() and 'sft' in args.pretrain.lower():
        model_type = 'phi3-sft'
    elif 'llama3-sft-v2' in args.pretrain.lower():
        model_type = 'rlhflow_llama_3_sft_8b_v2'
    elif 'llama-3.1-8b-instruct' in args.pretrain.lower():
        model_type = 'meta_llama_3_1_instruct_8b'
    else:
        model_type = 'debug'
        
    args.model_type = model_type
    
    args.wandb_run_name = f"{args.exp_prefix}_Ep{args.num_episodes}_{model_type}_fit_{args.reward_fit_dataset}_entropy-{args.entropy_threshold}_kl-{args.init_kl_coef}_value_clip-{args.value_clip}_ppo_reward_type-{args.ppo_reward_type}_{datetime.now().strftime('%Y%m%dT%H%M')}"

    train(args)
