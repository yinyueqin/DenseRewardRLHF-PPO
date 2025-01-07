import torch
from denserlhf.models import get_llm_for_sequence_regression
import torch.nn.functional as F
from typing import Dict

def build_denserlhf_rm(model_name, **kwargs):
    reward_model = get_llm_for_sequence_regression(model_name, "reward", **kwargs)
    reward_model.eval().requires_grad_(False)
    return reward_model


class DenserSegRMFPipeline:
    def __init__(self, task, model, ref_model, tokenizer):
        self.task = task
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.tokenizer.padding_size = "left"
        self.tokenizer.truncation_side = "left"
        self.max_prompt_length = 1728
        self.max_length = 2048
        self.label_pad_token_id = -100
        self.print_cnt_template = 0

    def tokenize_batch_element(
        self,
        prompt_text: str,
        response_text: str,
    ) -> Dict:
        batch = {}
        
        if self.print_cnt_template < 10:
            print(f"\nDebug prompt: {prompt_text}")
            print(f"\nDebug response: {response_text}")
            print('\nself.tokenizer.eos_token_id:', self.tokenizer.eos_token_id)
            print('\nCombined sequence:', prompt_text + response_text)
        
        self.print_cnt_template += 1     

        response_tokens = self.tokenizer(response_text, add_special_tokens=False)
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)


        if len(prompt_tokens["input_ids"]) + len(response_tokens["input_ids"]) > self.max_length:
            prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}

        if len(prompt_tokens["input_ids"]) + len(response_tokens["input_ids"]) > self.max_length:
            response_tokens = {k: v[: self.max_length - len(prompt_tokens["input_ids"])] for k, v in response_tokens.items()}

        if len(response_tokens["input_ids"]):
            if 'phi' in self.model_type.lower():
                response_tokens["input_ids"][-1] = 32007
                response_tokens["attention_mask"][-1] = True
            else:
                response_tokens["input_ids"][-1] = self.tokenizer.eos_token_id
                response_tokens["attention_mask"][-1] = True

        response_sequence_tokens = {k: prompt_tokens[k] + response_tokens[k] for k in response_tokens}

        response_sequence_tokens["labels"] = response_sequence_tokens["input_ids"][:]
        response_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )


        for k, toks in {
            "response": response_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        return batch

    def pad_left(self, sequence, max_length, padding_value):
        padding = [padding_value] * (max_length - len(sequence))
        return torch.tensor(padding + sequence)

    def entropy(self, logits):
        log_probs = F.log_softmax(logits, dim=-1)
        return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)


    def segment_data_peak(self, input_ids, attention_mask, entropy_thresholds, input_labels):
        batch_size, seq_len = input_ids.shape
        with torch.no_grad():
            out = self.ref_model(input_ids, attention_mask=attention_mask)
            logits = out.logits
            batch_entropy = self.entropy(logits).detach()

        entropy_thresholds = torch.tensor(entropy_thresholds, device=batch_entropy.device).repeat(batch_size, 1)
        mask = (batch_entropy.unsqueeze(1) > entropy_thresholds.unsqueeze(2)).float() 

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

               
                all_segments_end.append(segment_end_positions)

        return all_segments_end
    
    
    def __call__(self, samples_prompt, samples_response, entropy_thresholds, **kwargs):
        self.num_thresholds = kwargs.get("num_thresholds", 1)
        self.agg_func = kwargs.get("agg_func", "avg")
        self.segment_method = kwargs.get("segment_method", "peak")
        self.model_type = kwargs.get("model_type", "debug")

        tokenized_batch = [self.tokenize_batch_element(prompt, response) 
                        for prompt, response in zip(samples_prompt, samples_response)]

        response_ids, response_masks, response_labels = zip(*[(batch["response_input_ids"], 
                                                            batch["response_attention_mask"], 
                                                            batch["response_labels"]) 
                                                            for batch in tokenized_batch])

        max_length = max(len(seq) for seq in response_ids)
        response_ids = torch.stack([self.pad_left(seq, max_length, self.tokenizer.pad_token_id) for seq in response_ids]).to(torch.cuda.current_device())
        response_masks = torch.stack([self.pad_left(seq, max_length, 0) for seq in response_masks]).to(torch.cuda.current_device())
        response_labels = torch.stack([self.pad_left(seq, max_length, self.label_pad_token_id) for seq in response_labels]).to(torch.cuda.current_device())

        response_lengths = (response_labels != -100).sum(dim=1).cpu().numpy()

        segment_methods = {
            "peak": self.segment_data_peak,
        }
        segment_func = segment_methods.get(self.segment_method)
        if segment_func:
            all_segments_end = segment_func(response_ids, response_masks, entropy_thresholds, response_labels)
        else:
            print("Segment method not implemented")
            return

        with torch.no_grad():
            all_values_sequences, _ = self.model(response_ids, attention_mask=response_masks, return_output=True, return_every_step_reward=True)

        batch_size = all_values_sequences.shape[0]

        def calculate_segment_lengths(segment_end_positions, response_labels):
            if len(segment_end_positions) > 1:
                start = (response_labels != -100).nonzero(as_tuple=True)[0][0].item()
                first_segment_length = segment_end_positions[0] - start + 1
                subsequent_lengths = torch.diff(torch.tensor(segment_end_positions))
                return torch.cat((torch.tensor([first_segment_length]), subsequent_lengths.float()))
            else:
                return torch.tensor([(response_labels != -100).sum(-1).item()]).float()

        self.cnt = getattr(self, 'cnt', 0)

        all_segments_values = []
        all_segments_lengths = []
        all_original_segment_rewards = []
        all_segment_positions = []
        decoded_sentences = []
        segmented_sentences_with_rewards = []
        all_aggregated_values = []
        segment_num_list = []

        for i in range(batch_size):
            full_sentence = self.tokenizer.decode(response_ids[i][response_masks[i].bool()], skip_special_tokens=True)
            decoded_sentences.append(full_sentence)
            
            batch_segments_values = []
            batch_segments_lengths = []
            batch_segmented_sentence_with_reward = []
            
            for thresh in range(self.num_thresholds):
                segment_end_positions = all_segments_end[i * self.num_thresholds + thresh]
                segments_values = [all_values_sequences[i, end_pos] for end_pos in segment_end_positions]
                
                if self.cnt % 70 == 0:
                    print(f'\nprint segment_end_positions {segment_end_positions}, seq_len {len(response_ids[i])}')
                    print(f"\nFull response for batch {i}:\n{full_sentence}")
                    
                    start = (response_labels[i] != -100).nonzero(as_tuple=True)[0][0].item()
                    for idx, end in enumerate(segment_end_positions):
                        segment = response_ids[i][start:end+1]
                        decoded_segment = self.tokenizer.decode(segment, skip_special_tokens=True)
                        print(f"\nSegment {idx + 1}: {decoded_segment} | Reward: {segments_values[idx].item():.4f}")
                        start = end + 1
                
                self.cnt += 1

                num_segments = len(segments_values)
                segment_positions = [f"{(j+1)/num_segments:.3f}" for j in range(num_segments)]
                all_segment_positions.extend(segment_positions)
                all_original_segment_rewards.extend(segments_values)

                segment_lengths = calculate_segment_lengths(segment_end_positions, response_labels[i])
                
                start = (response_labels[i] != -100).nonzero(as_tuple=True)[0][0].item()
                for j, (_, end) in enumerate(zip(segment_lengths, segment_end_positions)):
                    segment_text = self.tokenizer.decode(response_ids[i][start:end+1], skip_special_tokens=True)
                    batch_segmented_sentence_with_reward.append((segment_text, segments_values[j]))
                    start = end + 1

                batch_segments_lengths.append(segment_lengths.mean())

                if "avg" in self.agg_func:
                    segment_value = torch.mean(torch.stack(segments_values))
                
                batch_segments_values.append(segment_value)
                segment_num_list.append(len(segments_values))

            all_segments_values.append(torch.stack(batch_segments_values))
            all_segments_lengths.append(torch.stack(batch_segments_lengths))
            segmented_sentences_with_rewards.append(batch_segmented_sentence_with_reward)
            all_aggregated_values.append(segment_value)

        outputs = torch.stack(all_segments_values)
        segment_lengths = torch.stack(all_segments_lengths)
        segment_num = torch.tensor(segment_num_list).reshape(batch_size, -1)
        response_length = torch.tensor(response_lengths)

        segment_position_rewards = [
            {pos: reward.item()} for pos, reward in zip(all_segment_positions, all_original_segment_rewards)
        ]

        additional_info = {
            "decoded_sentences": decoded_sentences,
            "segmented_sentences_with_rewards": segmented_sentences_with_rewards,
            "aggregated_values": all_aggregated_values
        }

        return outputs, segment_num, response_length, segment_lengths, all_original_segment_rewards, segment_position_rewards, additional_info