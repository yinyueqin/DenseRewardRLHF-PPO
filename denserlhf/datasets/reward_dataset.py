from typing import Callable
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import exist_and_not_none, zero_pad_sequences

from typing import Dict
from torch.nn.utils.rnn import pad_sequence
import re

def pad_left(sequence, max_length, padding_value):
    padding = [padding_value] * (max_length - len(sequence))
    return torch.tensor(padding + sequence)


class PreferenceDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        max_prompt_length: int,
        strategy,
        model_type: str,
        label_pad_token_id: int = -100
    ) -> None:
        super().__init__()
       
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.margins = []
        self.metadatas = []
        self.model_type = model_type
        self.label_pad_token_id = label_pad_token_id

        self.prompt_text = []
        self.complete_chosen_text = []
        self.complete_reject_text = []
        self.margins = []
        self.print_cnt = 0
        self.strategy.print('\ntokenizer.padding_side: ', self.tokenizer.padding_side)
        self.strategy.print('\ntokenizer.truncation_side: ', self.tokenizer.truncation_side)

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt = data["chosen"][:-1]
            self.prompt_text.append(prompt)
            self.complete_chosen_text.append(data["chosen"])
            self.complete_reject_text.append(data["rejected"])
            
            if "chosen-rating" in data and "rejected-rating" in data and isinstance(data["chosen-rating"], (int, float)) and isinstance(data["rejected-rating"], (int, float)):
                margin = data["chosen-rating"] - data["rejected-rating"]
            elif "chosen_score" in data and "rejected_score" in data and isinstance(data["chosen_score"], (int, float)) and isinstance(data["rejected_score"], (int, float)):
                margin = data["chosen_score"] - data["rejected_score"]
            else:
                margin = 0
            self.margins.append(margin)
            self.metadatas.append(data.get("metadata", {}))

        strategy.print("example data:")
        strategy.print("chosen_text:", self.complete_chosen_text[0])
        strategy.print("rejected_text:", self.complete_reject_text[0])
        strategy.print("margin:", self.margins[0])
        strategy.print("metadata:", self.metadatas[0])
    
    def __len__(self):
        length = len(self.complete_chosen_text)
        return length

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        idx: int,
    ) -> Dict:

        batch = {}

        chosen_text = self.tokenizer.apply_chat_template([chosen[-1]], tokenize=False, add_generation_prompt=False)
        rejected_text = self.tokenizer.apply_chat_template([rejected[-1]], tokenize=False, add_generation_prompt=False)
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
        
        elif self.model_type == "meta_llama_3_1_instruct_8b":
            marker = '<|eot_id|>'
            pos = chosen_text.find('<|eot_id|>')
            chosen_text = chosen_text[pos + len(marker):].replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
            rejected_text = rejected_text[pos + len(marker):].replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
            prompt_text = prompt_text[pos + len(marker):] + "<|start_header_id|>assistant<|end_header_id|>\n\n"

                

        chosen_tokens = self.tokenizer(chosen_text, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected_text, add_special_tokens=False)
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)


        if self.print_cnt % 1000 == 0:
            if 'phi' in self.model_type.lower():
                self.strategy.print('\nuse Phi3 chat template')
            elif '<|eot_id|>' in self.tokenizer.chat_template:
                self.strategy.print('\nuse Meta llama chat template')
            self.strategy.print('\nself.tokenizer.eos_token_id: ', self.tokenizer.eos_token_id)
            self.strategy.print('\nself.tokenizer.padding_side: ', self.tokenizer.padding_side)
            self.strategy.print('\nself.tokenizer.truncation_side: ', self.tokenizer.truncation_side)
            self.strategy.print('\nexample idx: ', idx)
            self.strategy.print('\ntokenized chosen content in tokenize_batch_element: ', chosen_text)
            self.strategy.print('\ntokenized rejected content in tokenize_batch_element: ', rejected_text)
            self.strategy.print('\ntokenized prompt_text in tokenize_batch_element: ', prompt_text)
            self.strategy.print('\nCombined chosen text', prompt_text + chosen_text)
            self.strategy.print('\nchosen_tokens: ', chosen_tokens)
            self.strategy.print('\nCombined rejected text', prompt_text + rejected_text)
            self.strategy.print('\nrejected_tokens: ', rejected_tokens)
            self.strategy.print('\nprompt_tokens: ', prompt_tokens)
            
        self.print_cnt += 1

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}

        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - len(prompt_tokens["input_ids"])] for k, v in chosen_tokens.items()}
            rejected_tokens = {
                k: v[: self.max_length - len(prompt_tokens["input_ids"])] for k, v in rejected_tokens.items()
            }
        
        if len(chosen_tokens["input_ids"]):
            if 'phi' in self.model_type.lower():
                chosen_tokens["input_ids"][-1] = 32007
                chosen_tokens["attention_mask"][-1] = True
            else:
                chosen_tokens["input_ids"][-1] = self.tokenizer.eos_token_id
                chosen_tokens["attention_mask"][-1] = True

        if len(rejected_tokens["input_ids"]):
            if 'phi' in self.model_type.lower():
                rejected_tokens["input_ids"][-1] = 32007
                rejected_tokens["attention_mask"][-1] = True
            else:
                rejected_tokens["input_ids"][-1] = self.tokenizer.eos_token_id
                rejected_tokens["attention_mask"][-1] = True

        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        return batch

    def __getitem__(self, idx):
        prompt_text, chosen_text, reject_text= self.prompt_text[idx], self.complete_chosen_text[idx], self.complete_reject_text[idx]
        batch_element = self.tokenize_batch_element(prompt_text, chosen_text, reject_text, idx)

        return (
            batch_element["chosen_input_ids"],
            batch_element["chosen_attention_mask"],
            batch_element['chosen_labels'],
            batch_element['rejected_input_ids'],
            batch_element['rejected_attention_mask'],
            batch_element['rejected_labels'],
            self.margins[idx],
            self.metadatas[idx]
        )

    def collate_fn(self, item_list):

        chosen_ids = []
        chosen_masks = []
        chosen_labels = []
        reject_ids = []
        rejects_masks = []
        rejects_labels = []
        margins = []
        metadatas = []
        for chosen_id, chosen_mask, chosen_label, reject_id, rejects_mask, rejects_label, margin, metadata in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            chosen_labels.append(chosen_label)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            rejects_labels.append(rejects_label)
            margins.append(margin)
            metadatas.append(metadata)

        max_length = max(max(len(seq) for seq in chosen_ids), 
                     max(len(seq) for seq in reject_ids))

        chosen_ids = torch.stack([pad_left(seq, max_length, self.tokenizer.pad_token_id) for seq in chosen_ids])
        chosen_masks = torch.stack([pad_left(seq, max_length, 0) for seq in chosen_masks])
        chosen_labels = torch.stack([pad_left(seq, max_length, self.label_pad_token_id) for seq in chosen_labels])
        reject_ids = torch.stack([pad_left(seq, max_length, self.tokenizer.pad_token_id) for seq in reject_ids])
        rejects_masks = torch.stack([pad_left(seq, max_length, 0) for seq in rejects_masks])
        rejects_labels = torch.stack([pad_left(seq, max_length, self.label_pad_token_id) for seq in rejects_labels])

        return chosen_ids, chosen_masks, chosen_labels, reject_ids, rejects_masks, rejects_labels, margins, metadatas

