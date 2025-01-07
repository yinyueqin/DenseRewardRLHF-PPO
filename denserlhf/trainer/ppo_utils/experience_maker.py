import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F

from denserlhf.models.actor import Actor
from denserlhf.models.utils import compute_reward, masked_mean, unpacking_samples
from denserlhf.utils.logging_utils import init_logger
from denserlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

import json
import os

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device)


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory()


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.values = to(self.values, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.values = pin_memory(self.values)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
        output_max_len=1024,
        entropy_threshold=2.0,
        num_thresholds=1,
        ppo_reward_type="segment_normalization",
        agg_func="avg",
        segment_method="peak",
        reward_fit_dataset='preference700k',
        model_type='phi3',
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.output_max_len = output_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.rm_input_max_len = prompt_max_len + output_max_len
        self.entropy_threshold = entropy_threshold
        self.num_thresholds = num_thresholds
        self.minimum_segment_length = 1
        self.ppo_reward_type = ppo_reward_type
        self.agg_func = agg_func
        self.segment_method = segment_method
        self.reward_fit_dataset = reward_fit_dataset
        self.model_type = model_type

        if 'end_penalty' in self.strategy.args.exp_prefix:
            if 'm0d05' in self.strategy.args.exp_prefix:
                self.penalty = -0.05
            else:
                self.penalty = 0
        
            self.strategy.print(f"\nself.penalty: {self.penalty}")
        
        if 'len_thres' in self.strategy.args.exp_prefix:
            if '800' in self.strategy.args.exp_prefix:
                self.response_penalty_length = 800
            else:
                self.response_penalty_length = output_max_len
            
            self.strategy.print(f"\nself.response_penalty_length: {self.response_penalty_length}")

        self.verbose = True
        self.print_position_cnt = 0
        self.print_tokenizer_info = 0        


    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if self.print_tokenizer_info == 0:
            self.strategy.print('\nself.tokenizer.padding_side: {}'.format(self.tokenizer.padding_side))
            self.strategy.print('\nself.tokenizer.truncation_side: {}'.format(self.tokenizer.truncation_side))
            self.tokenizer.truncation_side = 'left'
            self.strategy.print('\nself.tokenizer.truncation_side: {}'.format(self.tokenizer.truncation_side))
            self.print_tokenizer_info += 1
        if not padding:
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def get_generations(self, prompts, **generate_kwargs):
        if isinstance(prompts[0], dict):
            prompt_strs = [x['prompt'] for x in prompts]
            inputs = self.tokenize_fn(prompt_strs, self.prompt_max_len, device="cuda")
        else:
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask, output_seqs, eos_token_present = self.actor.generate(**inputs, **generate_kwargs)
        return sequences, attention_mask, action_mask, output_seqs, eos_token_present
    
    def segment_data_peak(self, input_ids, action_mask, entropy_thresholds, batch_entropy):
        batch_size, seq_len = input_ids.shape

        entropy_thresholds = torch.tensor(entropy_thresholds, device=batch_entropy.device).repeat(batch_size, 1)
        mask = (batch_entropy.unsqueeze(1) > entropy_thresholds.unsqueeze(2)).float()

        all_segments_end = []

        for i in range(batch_size):
            for thresh in range(self.num_thresholds):
                segment_end_positions = []
                current_segment_length = 0

                eos_index = action_mask[i].nonzero()[-1].item()

                for j in range(0, eos_index + 1):
                    current_segment_length += 1

                    if mask[i, thresh, j].item() == 1 and (current_segment_length) >= self.minimum_segment_length:
                        segment_end_positions.append(j)
                        current_segment_length = 0

                if entropy_thresholds[i].item() != 1000:
                    if len(segment_end_positions) == 0:
                        if seq_len > 1:
                            segment_end_positions.append(eos_index - 1)
                        else:
                            segment_end_positions.append(eos_index)
                    else:
                        if segment_end_positions[-1] == eos_index:
                            if len(segment_end_positions) > 1 and segment_end_positions[-2] == eos_index - 1:
                                segment_end_positions.pop()
                            else:
                                segment_end_positions[-1] = eos_index - 1
                        elif segment_end_positions[-1] < eos_index - 1:
                            segment_end_positions.append(eos_index - 1)
                else:
                    if len(segment_end_positions) == 0 or segment_end_positions[-1] != eos_index:
                        segment_end_positions.append(eos_index)

                all_segments_end.append(segment_end_positions)

        return all_segments_end

    def get_adjusted_mean_std(self, position_percentage):

        def convert_keys_to_float(obj):
            if isinstance(obj, dict):
                return {float(k) if k.replace('.', '', 1).isdigit() else k: convert_keys_to_float(v)
                        for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys_to_float(i) for i in obj]
            else:
                return obj

        if not hasattr(self, '_cached_params'):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            param_path = os.path.join(current_dir, self.strategy.args.norm_params_path)
            with open(param_path, 'r') as f:
                self._cached_params = convert_keys_to_float(json.load(f))
    
        p = (self._cached_params.get(self.model_type, {})
             .get(self.reward_fit_dataset, {})
             .get(self.segment_method, {})
             .get(self.agg_func, {})
             .get(self.entropy_threshold, {}))
        
        position_percentage = torch.tensor(position_percentage)

        if p:
            adjusted_mean = p[0] * torch.log(position_percentage + 1e-10) + p[2]
            adjusted_std = p[1] * torch.log(position_percentage + 1e-10) + p[3]

            if self.verbose:
                self.strategy.print(f"\nModel type: {self.model_type}")
                self.strategy.print(f"\nentropy_threshold: {self.entropy_threshold}")
                self.strategy.print(f"\nreward_fit_dataset: {self.reward_fit_dataset}")
                self.strategy.print(f"\nsegment_method: {self.segment_method}")
                self.strategy.print(f"\nagg_func: {self.agg_func}")
                self.strategy.print(f"\nmean function: {p[0]} * log(position_percentage) + {p[2]}")
                self.strategy.print(f"\nstd function: {p[1]} * log(position_percentage) + {p[3]}")
                self.verbose = False
                
            return adjusted_mean, adjusted_std
        
        return 0, 1

    def get_segment_rewards(self, sequences, attention_mask, action_mask, eos_token_present, device, base_action_entropy=None):
        with torch.no_grad():
            entropy_thresholds = [self.entropy_threshold]
            rm_sequence, rm_attn_mask = sequences, attention_mask
            num_actions = action_mask.size(1)
            prompt_len = rm_sequence.size(1) - num_actions
            
            if self.segment_method == "peak":
                segment_note = 'peak'
                all_segments_end = self.segment_data_peak(rm_sequence, action_mask, entropy_thresholds, base_action_entropy)

            all_values_sequences, _ = self.reward_model(
                rm_sequence, 
                attention_mask=rm_attn_mask, 
                return_output=True, 
                return_every_step_reward=True
            )
            
        
            all_values_sequences = all_values_sequences[:, prompt_len:]

            batch_size = rm_sequence.size(0)

            if self.ppo_reward_type not in ["segment_last_avg", "segment_last_logsumexp"]:
                rewards = torch.zeros_like(action_mask, dtype=torch.float)
            else:
                rewards = torch.zeros(batch_size, dtype=torch.float)
            rewards = rewards.to(device)

            all_segments_lengths = []
        
            sample_segment_lengths = [[] for _ in range(batch_size)]
            sample_segment_rewards = [[] for _ in range(batch_size)]

            for i in range(batch_size):
                segment_end_positions = [end_pos for end_pos in all_segments_end[i]]
                segment_values = []
                segment_lengths = []
                temp_lengths_for_averaging = []
                last_eos_index = (action_mask[i].nonzero().max().item() 
                                  if action_mask[i].nonzero().size(0) > 0 else 0)
                
                if self.print_position_cnt % 100 == 0:
                    self.strategy.print(
                        f"\nsegment_end_positions: {segment_end_positions}, "
                        f"last_eos_index: {last_eos_index}, prompt_len: {prompt_len}"
                    )
                    
                    full_sequence = self.tokenizer.decode(rm_sequence[i], skip_special_tokens=True)
                    self.strategy.print(f"\nFull sequence for batch {i}:\n{full_sequence}")
                    
                    prev_end = prompt_len
                    for idx, end in enumerate(segment_end_positions):
                        sequences_end = prompt_len + end
                        segment = rm_sequence[i][prev_end:sequences_end+1]
                        decoded_segment = self.tokenizer.decode(segment, skip_special_tokens=True)
                        segment_reward = all_values_sequences[i, end].item()
                        segment_length = sequences_end - prev_end + 1
                        self.strategy.print(
                            f"\nSegment {idx + 1}: {decoded_segment} | "
                            f"Reward: {segment_reward:.4f} | Length: {segment_length}"
                        )
                        prev_end = sequences_end + 1

                self.print_position_cnt += 1
                    
                for pos, end_pos in enumerate(segment_end_positions):
                    value_at_end = all_values_sequences[i, end_pos]

                    if self.reward_model.normalize_reward:
                        if self.ppo_reward_type in ["segment_normalization", "same_segment_last", "every_segment_last"]:
                            if 'no_normalization' in self.strategy.args.exp_prefix:
                                adjusted_mean, adjusted_std = 0, 1
                            else:
                                total_segments = len(segment_end_positions)
                                position_percentage = (pos + 1) / total_segments
                                adjusted_mean, adjusted_std = self.get_adjusted_mean_std(position_percentage)
                            value_at_end = (value_at_end - adjusted_mean) / adjusted_std
                        else:
                            # bandit rm
                            value_at_end = (value_at_end - self.reward_model.mean) / self.reward_model.std
                            if self.verbose:
                                print(f"self.ppo_reward_type: {self.ppo_reward_type}, "
                                      f"use one Mean: {self.reward_model.mean}, std: {self.reward_model.std}")
                                print(f"\nsegment_note: {segment_note}")
                                self.verbose = False
                    else:
                        if self.verbose:
                            print(f"self.ppo_reward_type: {self.ppo_reward_type}, no normalization")
                            self.verbose = False

                    start_pos = segment_end_positions[pos-1] + 1 if pos > 0 else 0
                    segment_length = end_pos - start_pos + 1
                    
                    sample_segment_lengths[i].append(segment_length)
                    sample_segment_rewards[i].append(value_at_end.item())
                    
                    temp_lengths_for_averaging.append(segment_length)
                    segment_values.append(value_at_end)
                    segment_lengths.append(segment_length)
                
                if self.ppo_reward_type in ["every_segment_last", "same_segment_last", "segment_normalization"]:
                    rewards_matrix = rewards[i]

                    if 'end_penalty' not in self.strategy.args.exp_prefix:
                        eos_token_present[i] = True

                    response_length = last_eos_index + 1 - prompt_len
                    if (hasattr(self, 'response_penalty_length') 
                        and response_length > self.response_penalty_length):
                        rewards_matrix[:] = self.penalty
                    else:
                        if eos_token_present[i]:
                            for pos, (value, end_pos) in enumerate(zip(segment_values, segment_end_positions)):
                                start_pos = segment_end_positions[pos-1] + 1 if pos > 0 else 0
                                if self.ppo_reward_type == "every_segment_last":
                                    rewards_matrix[end_pos] = value
                                elif self.ppo_reward_type == "same_segment_last":
                                    rewards_matrix[start_pos:end_pos+1] = value
                                elif self.ppo_reward_type in ["segment_normalization"]:
                                    segment_length = end_pos - start_pos + 1
                                    normalized_value = value / segment_length
                                    if self.entropy_threshold == 0 and 'length_norm' in self.strategy.args.exp_prefix:
                                        response_length = last_eos_index + 1
                                        normalized_value = normalized_value / response_length
                                    rewards_matrix[start_pos:end_pos+1] = normalized_value

                            if last_eos_index + 1 < num_actions:
                                rewards_matrix[last_eos_index + 1:] = 0
                        else:
                            rewards_matrix[:] = self.penalty
                                
                elif self.ppo_reward_type == "segment_last_logsumexp":
                    if 'end_penalty' not in self.strategy.args.exp_prefix:
                        eos_token_present = True

                    response_length = last_eos_index + 1 - prompt_len
                    if (hasattr(self, 'response_penalty_length') 
                        and response_length > self.response_penalty_length):
                        rewards[i] = self.penalty
                        continue

                    if eos_token_present:
                        rewards[i] = (self.log_sum_exp_temperature 
                                       * torch.logsumexp(torch.tensor(segment_values) 
                                                         / self.log_sum_exp_temperature, dim=0))
                    else:
                        rewards[i] = self.penalty

                elif self.ppo_reward_type == "segment_last_avg":
                    if 'end_penalty' not in self.strategy.args.exp_prefix:
                        eos_token_present = True

                    response_length = last_eos_index + 1 - prompt_len
                    if (hasattr(self, 'response_penalty_length') 
                        and response_length > self.response_penalty_length):
                        rewards[i] = self.penalty
                        continue

                    if eos_token_present:
                        rewards[i] = torch.tensor(segment_values).mean()
                    else:
                        rewards[i] = self.penalty
                
                average_lengths = torch.tensor(temp_lengths_for_averaging).float().mean()
                all_segments_lengths.append(average_lengths)

            
            if self.strategy.args.print_index % 4 == 0:
                for sample_idx in range(2):
                    reward_length_pairs = list(zip(sample_segment_rewards[sample_idx], 
                                                   sample_segment_lengths[sample_idx]))
                    self.strategy.print(
                        f"\nSample {sample_idx + 1} segments (normalized reward, length): {reward_length_pairs}"
                    )

            avg_lengths = torch.stack(all_segments_lengths)

            return rewards, avg_lengths

    def entropy(self, logits):
        log_probs = F.log_softmax(logits, dim=-1)
        return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs):
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        sequences, attention_mask, action_mask, output_seqs, eos_token_present = self.get_generations(prompts, **generate_kwargs)
        num_actions = action_mask.size(1)

        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        base_action_log_probs, base_logits = self.initial_model(
            sequences, num_actions, attention_mask, return_output=True
        )

        logits_slice = base_logits["logits"][:, -num_actions:].clone()
        
        base_action_entropy = self.entropy(logits_slice)

        # values
        value = self.critic(sequences, num_actions, attention_mask)

        r, avg_segment_lengths = self.get_segment_rewards(
            sequences, attention_mask, action_mask, eos_token_present,
            device="cuda", base_action_entropy=base_action_entropy
        )

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            ppo_reward_type=self.ppo_reward_type,
        )

        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "reward_minus_kl": reward,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
            "avg_segment_lengths": avg_segment_lengths,
        }
        
        # reset model state
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

