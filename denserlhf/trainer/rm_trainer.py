import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from denserlhf.models import LogExpLoss, PairWiseLoss
from denserlhf.utils.distributed_sampler import DistributedSampler

import torch.nn.functional as F

class RewardModelTrainer(ABC):
    """
    Trainer for training a reward model.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to apply.
        optim (Optimizer): The optimizer to use during training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler for dynamic adjustments during training.
        tokenizer (Tokenizer): The tokenizer for processing input text data.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
        loss (str, defaults to "sigmoid"): The loss function to use during training, e.g., "sigmoid".
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
        ref_model=None,
        entropy_threshold=2,
        num_thresholds=1,
        agg_func="avg",
        segment_method="peak",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.ref_model = ref_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.entropy_threshold = entropy_threshold
        self.num_thresholds = num_thresholds
        self.agg_func = agg_func
        self.segment_method = segment_method
        self.args = strategy.args
        self.cnt = 0

        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss")
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss")

        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.packing_samples = strategy.args.packing_samples

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            self.ref_model.eval()
            acc_mean = 0
            loss_mean = 0
            for chosen_ids, c_mask, chosen_labels, reject_ids, r_mask, reject_labels, margin, _ in self.train_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                chosen_labels = chosen_labels.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                reject_labels = reject_labels.squeeze(1).to(torch.cuda.current_device())
               
                chosen_response_lengths = (chosen_labels != -100).sum(dim=1).cpu().numpy()
                reject_response_lengths = (reject_labels != -100).sum(dim=1).cpu().numpy()
                

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                
                chosen_reward, reject_reward, chosen_segment_num, reject_segment_num, chosen_segment_length, reject_segment_length, aux_loss = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, chosen_labels, reject_ids, r_mask, reject_labels
                )
                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()
                if margin is not None:
                    margin = torch.tensor(margin, dtype=torch.float32).to(torch.cuda.current_device())  # Ensure margin is a float tensor
                    margin = margin.unsqueeze(1).repeat(1, self.num_thresholds).view(-1)  # Repeat margin for each threshold
                preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)

                if not self.aux_loss:
                    aux_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_mean = acc_mean * 0.9 + 0.1 * acc
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                logs_dict = {
                    "preference_loss": preference_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    'reward_difference': chosen_reward.mean().item() - reject_reward.mean().item(),
                    "chosen_segment_num": chosen_segment_num.mean().item(),
                    "reject_segment_num": reject_segment_num.mean().item(),
                    'chosen_response_lengths': chosen_response_lengths.mean().item(),
                    'reject_response_lengths': reject_response_lengths.mean().item(),
                    "chosen_segment_length": chosen_segment_length.mean().item(),
                    "reject_segment_length": reject_segment_length.mean().item(),
                    "loss_mean": loss_mean,
                    "acc_mean": acc_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )
            ckpt_model_path = os.path.join(args.save_path, f"ckpt_{global_step}")
            self.strategy.save_model(self.model, self.tokenizer, ckpt_model_path)
            
    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            acc_count = 0
            sample_count = 0
            chosen_rewards = []
            rejected_rewards = []
            chosen_segment_num = []
            reject_segment_num = []
            chosen_segment_length = []
            reject_segment_length = []
            chosen_response_lengths = []
            reject_response_lengths = []
            
            loss_sum = 0
            for chosen_ids, c_mask, chosen_labels, reject_ids, r_mask, reject_labels, margin, metadatas in eval_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                chosen_labels = chosen_labels.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                reject_labels = reject_labels.squeeze(1).to(torch.cuda.current_device())

                batch_chosen_response_lengths = (chosen_labels != -100).sum(dim=1).cpu().numpy()
                batch_reject_response_lengths = (reject_labels != -100).sum(dim=1).cpu().numpy()

                chosen_response_lengths.append(torch.tensor(batch_chosen_response_lengths).flatten())
                reject_response_lengths.append(torch.tensor(batch_reject_response_lengths).flatten())


                margin = torch.tensor(margin).to(torch.cuda.current_device())

                if self.margin_loss and margin is not None:
                    margin = torch.tensor(margin, dtype=torch.float32).to(torch.cuda.current_device())
                    margin = margin.unsqueeze(1).repeat(1, self.num_thresholds).view(-1)
                else:
                    margin = None

                chosen_reward, reject_reward, batch_chosen_segment_num, batch_reject_segment_num, \
                batch_chosen_response_length, batch_reject_response_length, _ = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, chosen_labels, reject_ids, r_mask, reject_labels
                )
                
                if margin is not None:
                    margin = torch.tensor(margin, dtype=torch.float32).to(torch.cuda.current_device())
                    margin = margin.unsqueeze(1).repeat(1, self.num_thresholds).view(-1)


                loss = self.loss_fn(chosen_reward, reject_reward, margin)
                
                chosen_rewards.append(chosen_reward.flatten())
                rejected_rewards.append(reject_reward.flatten())
                
                chosen_segment_num.append(batch_chosen_segment_num.flatten())
                reject_segment_num.append(batch_reject_segment_num.flatten())
                
                chosen_segment_length.append(batch_chosen_response_length.flatten())
                reject_segment_length.append(batch_reject_response_length.flatten())
                
                acc_count += (chosen_reward > reject_reward).float().sum().item()
                sample_count += chosen_reward.shape[0]
                
                loss_sum += loss.item()
                step_bar.update()

            acc_mean = acc_count / sample_count
            loss_mean = loss_sum / eval_dataloader.__len__()

            chosen_rewards = torch.cat(chosen_rewards).float()
            chosen_rewards = self.strategy.all_gather(chosen_rewards)
            chosen_segment_num = torch.cat(chosen_segment_num)
            chosen_segment_num = self.strategy.all_gather(chosen_segment_num)
            chosen_segment_length = torch.cat(chosen_segment_length)
            chosen_segment_length = self.strategy.all_gather(chosen_segment_length)
            chosen_response_lengths = torch.cat(chosen_response_lengths).float()
            chosen_response_lengths = self.strategy.all_gather(chosen_response_lengths)


            rejected_rewards = torch.cat(rejected_rewards).float()
            rejected_rewards = self.strategy.all_gather(rejected_rewards)
            reject_segment_num = torch.cat(reject_segment_num)
            reject_segment_num = self.strategy.all_gather(reject_segment_num)
            reject_segment_length = torch.cat(reject_segment_length)
            reject_segment_length = self.strategy.all_gather(reject_segment_length)
            reject_response_lengths = torch.cat(reject_response_lengths).float()
            reject_response_lengths = self.strategy.all_gather(reject_response_lengths)

            all_rewards = torch.cat([chosen_rewards, rejected_rewards])
            reward_mean = torch.mean(all_rewards)
            reward_std = torch.std(all_rewards).clamp(min=1e-8)

            self.strategy.print("Set reward mean std")
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
                "chosen_reward_mean": chosen_rewards.mean().item(),
                "rejected_reward_mean": rejected_rewards.mean().item(),
                "chosen_segment_num": chosen_segment_num.mean().item(),
                "reject_segment_num": reject_segment_num.mean().item(),
                "chosen_segment_length": chosen_segment_length.mean().item(),
                "reject_segment_length": reject_segment_length.mean().item(),
                "chosen_response_length": chosen_response_lengths.mean().item(),
                "reject_response_length": reject_response_lengths.mean().item()
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            histgram = torch.histogram(all_rewards.cpu(), bins=10, range=(-10, 10), density=True)[0] * 2
            self.strategy.print("histgram")
            self.strategy.print(histgram)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()

    def entropy(self, logits):
        log_probs = F.log_softmax(logits, dim=-1)
        return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)


    def segment_data_peak(self, input_ids, attention_mask, entropy_thresholds, input_labels):
        batch_size, seq_len = input_ids.shape
        with torch.no_grad():
            out = self.ref_model(input_ids, attention_mask=attention_mask, return_output=True)
            logits = out.logits
            batch_entropy = self.entropy(logits).detach()  

        entropy_thresholds = torch.tensor(entropy_thresholds, device=batch_entropy.device).repeat(batch_size, 1)
        mask = (batch_entropy.unsqueeze(1) > entropy_thresholds.unsqueeze(2)).float()  

        del batch_entropy
        torch.cuda.empty_cache()

        all_segments_end = []

        for i in range(batch_size):
            for thresh in range(self.num_thresholds):
                segment_end_positions = []
                current_segment_length = 0

                non_negative_indices = (input_labels[i] != -100).nonzero(as_tuple=True)[0]
                if len(non_negative_indices) == 0:
                    start_index = seq_len - 1
                else:
                    start_index = non_negative_indices.min().item()

                for j in range(start_index, seq_len):
                    current_segment_length += 1

                    if mask[i, thresh, j].item() == 1:
                        segment_end_positions.append(j)
                        current_segment_length = 0

                if entropy_thresholds[i, thresh].item() != 1000:
                    if len(segment_end_positions) == 0:
                        if seq_len > 1:
                            segment_end_positions.append(seq_len - 2)
                        else:
                            segment_end_positions.append(seq_len - 1)
                    else:
                        if segment_end_positions[-1] == seq_len - 1:
                            if len(segment_end_positions) > 1 and segment_end_positions[-2] == seq_len - 2:
                                segment_end_positions.pop()
                            else:
                                segment_end_positions[-1] = seq_len - 2
                        elif segment_end_positions[-1] < seq_len - 2:
                            segment_end_positions.append(seq_len - 2)
                else:
                    if len(segment_end_positions) == 0 or segment_end_positions[-1] != seq_len - 1:
                        segment_end_positions.append(seq_len - 1)

                # if self.strategy.is_rank_0():
                #     if self.cnt % 7000 == 0:
                #         print('\n print segment_end_positions ', segment_end_positions, 'seq_len ', seq_len)
                        
                #         full_response = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                #         print(f"\nFull response for batch {i}:\n{full_response}")
                        
                #         prev_end = start_index
                #         for idx, end in enumerate(segment_end_positions):
                #             segment = input_ids[i][prev_end:end+1]
                #             decoded_segment = self.tokenizer.decode(segment, skip_special_tokens=True)
                #             print(f"\n Segment {idx + 1}: {decoded_segment}")
                #             prev_end = end + 1
                    
                #     self.cnt += 1
                all_segments_end.append(segment_end_positions)

        return all_segments_end
    

    def concatenated_forward(self, model, chosen_ids, c_mask, chosen_labels, reject_ids, r_mask, rejected_labels):
        input_ids, att_masks, input_labels = self.concatenated_inputs(chosen_ids, c_mask, chosen_labels, reject_ids, r_mask, rejected_labels)

        entropy_thresholds = [self.entropy_threshold for _ in range(self.num_thresholds)]

        if self.segment_method == "peak":
            all_segments_end = self.segment_data_peak(input_ids, att_masks, entropy_thresholds, input_labels)
        else:
            print("Segment method not implemented")
        batch_size = chosen_ids.size(0) * 2  
        all_values_sequences, output = model(input_ids, attention_mask=att_masks, return_output=True, return_every_step_reward=True)



        all_segments_values = []
        all_segments_lengths = []
        chosen_avg_segment_counts = []
        rejected_avg_segment_counts = []

        for i in range(batch_size):
            for thresh in range(self.num_thresholds):
                segment_end_positions = all_segments_end[i * self.num_thresholds + thresh]
                batch_segments_values = [all_values_sequences[i, end_pos] for end_pos in segment_end_positions]

                if self.strategy.is_rank_0():
                    if self.cnt % 200 == 0:
                        print('\n print segment_end_positions ', segment_end_positions, 'seq_len ', len(input_ids[i]))

                        print(f"\nLast token id: {input_ids[i][-1].item()}")
                        print(f"\nSecond last token id: {input_ids[i][-2].item()}")
                        
                        full_response = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                        print(f"\nFull response for batch {i}:\n{full_response}")
                        
                        start_index = (input_labels[i] != -100).nonzero(as_tuple=True)[0][0].item()
                        prev_end = start_index
                        for idx, end in enumerate(segment_end_positions):
                            segment = input_ids[i][prev_end:end+1]
                            decoded_segment = self.tokenizer.decode(segment, skip_special_tokens=True)
                            segment_reward = batch_segments_values[idx].item()
                            print(f"\nSegment {idx + 1}: {decoded_segment} | Reward: {segment_reward:.4f}")
                            prev_end = end + 1

                            if idx == len(segment_end_positions) - 1:
                                print(f"\nLast segment end token id: {input_ids[i][end].item()}") 
                    
                    self.cnt += 1

                if len(segment_end_positions) > 1:
                    first_segment_length = segment_end_positions[0] - ((input_labels[i] != -100).nonzero(as_tuple=True)[0][0].item() - 1)
                    subsequent_segment_lengths = torch.tensor(segment_end_positions[1:]) - torch.tensor(segment_end_positions[:-1])
                    segment_lengths = torch.cat((torch.tensor([first_segment_length]), subsequent_segment_lengths.float()))
                else:
                    segment_lengths = torch.tensor([(input_labels[i] != -100).sum(-1).item()]).float()
                
                all_segments_lengths.append(segment_lengths.mean())

                if self.agg_func == "avg":
                    segment_value = torch.stack(batch_segments_values).mean()
                else:
                    print("Aggregation function not implemented")
               
                all_segments_values.append(segment_value)

                if i < batch_size // 2:
                    chosen_avg_segment_counts.append(len(segment_end_positions))
                else:
                    rejected_avg_segment_counts.append(len(segment_end_positions))

        all_segments_values = torch.stack(all_segments_values)
        chosen_rewards = all_segments_values[:self.num_thresholds * batch_size // 2]
        rejected_rewards = all_segments_values[self.num_thresholds * batch_size // 2:]

        all_segments_lengths = torch.stack(all_segments_lengths).float()
        chosen_segment_lengths = all_segments_lengths[:self.num_thresholds * batch_size // 2].view(-1, self.num_thresholds).mean(dim=1)
        rejected_segment_lengths = all_segments_lengths[self.num_thresholds * batch_size // 2:].view(-1, self.num_thresholds).mean(dim=1)

        chosen_avg_segment_counts = torch.tensor(chosen_avg_segment_counts).float().view(-1, self.num_thresholds).mean(dim=1)
        rejected_avg_segment_counts = torch.tensor(rejected_avg_segment_counts).float().view(-1, self.num_thresholds).mean(dim=1)

        aux_loss = output.aux_loss if "aux_loss" in output else []

        return (
            chosen_rewards,
            rejected_rewards,
            chosen_avg_segment_counts,
            rejected_avg_segment_counts,
            chosen_segment_lengths,
            rejected_segment_lengths,
            aux_loss
        )

    def concatenated_inputs(self, chosen_ids, c_mask, chosen_labels, reject_ids, r_mask, rejected_labels):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)

        max_length = max(chosen_labels.shape[1], rejected_labels.shape[1])
        input_labels = torch.cat(
            (
                pad_to_length(chosen_labels, max_length, -100),
                pad_to_length(rejected_labels, max_length, -100),
            ),
            dim=0,
        )

        return inputs_ids, att_masks, input_labels
    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens):
        all_values, output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        half_len = len(packed_seq_lens) // 2
        chosen_rewards = all_values[:half_len]
        rejected_rewards = all_values[half_len:]
        aux_loss = output.aux_loss if "aux_loss" in output else []

        return chosen_rewards, rejected_rewards, aux_loss
