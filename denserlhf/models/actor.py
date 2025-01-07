from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import time
from .ring_attn_utils import convert_ring_attn_params
from .utils import log_probs_from_logits, reset_position_ids

def retry(pretrain_or_model, attn_implementation, quantization_config, bf16, device_map, retries=3, delay=30):
    for attempt in range(retries):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )
            return model
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        strategy=None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.counter = 0
        self.strategy = strategy

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

        
            self.model = retry(
                pretrain_or_model,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                bf16=bf16,
                device_map=device_map,
            )
            

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
            "tokenizer": kwargs.get("tokenizer", None)
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        if generate_args["num_beams"] > 1:
            generate_args["early_stopping"] = True
        else:
            generate_args["early_stopping"] = False

        tokenizer = generate_args["tokenizer"]
        if tokenizer is not None:
            if self.counter == 0:
                self.strategy.print(f"Current tokenizer padding side: {tokenizer.padding_side}")
            tokenizer.padding_side = 'left'
            if self.counter == 0:
                self.strategy.print(f"Updated tokenizer padding side: {tokenizer.padding_side}")
                self.strategy.print(f"num_beams: {generate_args['num_beams']}")
                self.strategy.print(f"early_stopping: {generate_args.get('early_stopping', 'Not set')}")

        if tokenizer.chat_template:
            if "<|end|>" in tokenizer.chat_template:
                generate_args["eos_token_id"] = [32000, 32007]
            elif self.strategy.args.model_type == "rlhflow_llama_3_sft_8b_v2":
                generate_args["eos_token_id"] = [128009, 128001]
            elif self.strategy.args.model_type == "meta_llama_3_1_instruct_8b":
                generate_args["eos_token_id"] = [128009, 128001, 128008]

        # Call generate
        try:
            sequences = self.model.generate(**generate_args)
        except RuntimeError as e:
            self.strategy.print(f"Error in generate: {e}")
            self.strategy.print(f"Generate args: {generate_args}")
            self.strategy.print(f"Model config: {self.model.config}")
            raise

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        # phi3
        if tokenizer.chat_template is not None and "<|end|>" in tokenizer.chat_template:
            end_token_id = generate_args["tokenizer"].encode("<|end|>")[0]
            if self.counter == 0:
                self.strategy.print('\n For Phi3, we have end_token_id:')
                self.strategy.print('\neos_token_id:', eos_token_id)
                self.strategy.print('\npad_token_id:', pad_token_id)
                self.strategy.print('\nend_token_id:', end_token_id)
                self.strategy.print('\ngenerate_args:', generate_args)
            self.counter += 1
            return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id, end_token_id)
        else:
            if self.counter == 0:
                self.strategy.print('\n For not Phi3, we do not have end_token_id:')
                self.strategy.print('\neos_token_id:', eos_token_id)
                self.strategy.print('\npad_token_id:', pad_token_id)
                self.strategy.print('\ngenerate_args:', generate_args)
            self.counter += 1
            return self.process_sequences_no_end_token(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id, end_token_id):
        if not isinstance(eos_token_id, list):
            eos_token_id = [eos_token_id]

        attention_mask = torch.ones_like(sequences, dtype=torch.long)
        for eos_id in eos_token_id:
            attention_mask &= sequences.ne(eos_id)
        attention_mask &= sequences.ne(pad_token_id)
        attention_mask = attention_mask.to(dtype=torch.long)

        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)

        generated_tokens = sequences[:, input_len:]
        
        # Track which samples contain EOS token in generated tokens
        eos_token_present = []
        for sample in generated_tokens:
            sample_contains_eos = any(token in eos_token_id for token in sample)
            if 'end_penalty' not in self.strategy.args.exp_prefix:
                sample_contains_eos = True
            eos_token_present.append(sample_contains_eos)

        condition = torch.gather(sequences, 1, eos_indices-1) != end_token_id
        attention_mask.scatter_(1, eos_indices, torch.where(condition, 1, 0))
        sequences.scatter_(1, eos_indices, torch.where(condition, end_token_id, eos_token_id[0]))

        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = torch.ones_like(state_seq, dtype=torch.bool)
        for eos_id in eos_token_id:
            action_mask &= state_seq.ne(eos_id)
        action_mask &= state_seq.ne(pad_token_id) & state_seq.ne(end_token_id)

        if self.counter % 100 == 0:
            self.strategy.print(f"\nattention_mask[0]: {attention_mask[0]}")
            self.strategy.print(f"\nstate_seq[0]: {state_seq[0]}")
            self.strategy.print(f"\naction_mask[0]: {action_mask[0]}")
            self.strategy.print(f"\nsequences[0]: {sequences[0]}")
            self.strategy.print(f"\neos_token_present: {eos_token_present}")


        return sequences, attention_mask, action_mask, sequences[:, input_len:], eos_token_present

    def process_sequences_no_end_token(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        if not isinstance(eos_token_id, list):
            eos_token_id = [eos_token_id]

        attention_mask = torch.ones_like(sequences, dtype=torch.long)
        for eos_id in eos_token_id:
            attention_mask &= sequences.ne(eos_id)
        attention_mask &= sequences.ne(pad_token_id)
        attention_mask = attention_mask.to(dtype=torch.long)

        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)

        generated_tokens = sequences[:, input_len:]
        
        eos_token_present = []
        for sample in generated_tokens:
            sample_contains_eos = any(token in eos_token_id for token in sample)
            if 'end_penalty' not in self.strategy.args.exp_prefix:
                sample_contains_eos = True
            eos_token_present.append(sample_contains_eos)

        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id[0])

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)

        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)

        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices[0]).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = torch.ones_like(state_seq, dtype=torch.bool)
        for eos_id in eos_token_id:
            action_mask &= state_seq.ne(eos_id)

        action_mask &= state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        if self.counter % 10 == 0:
            self.strategy.print(f"\nattention_mask[0]: {attention_mask[0]}")
            self.strategy.print(f"\nstate_seq[0]: {state_seq[0]}")
            self.strategy.print(f"\naction_mask[0]: {action_mask[0]}")
            self.strategy.print(f"\nsequences[0]: {sequences[0]}")
            self.strategy.print(f"\neos_token_present: {eos_token_present}")

        return sequences, attention_mask, action_mask, sequences[:, input_len:], eos_token_present

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if not self.packing_samples:
            position_ids = attention_mask.long().cumsum(-1) - 1
        else:
            if ring_attn_group is not None:
                sequences, attention_mask, position_ids = convert_ring_attn_params(
                    sequences, attention_mask, packed_seq_lens, ring_attn_group
                )
            else:
                # reset the positions for packed samples
                position_ids = reset_position_ids(attention_mask)
        position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)

        if num_actions is None:
            assert return_output
            return output

        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if not self.packing_samples:
            action_log_probs = log_probs[:, -num_actions:]
        else:
            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_log_probs.append(log_probs[:, start:end])
                offset += seq_len
            action_log_probs = torch.cat(action_log_probs, dim=1)

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
