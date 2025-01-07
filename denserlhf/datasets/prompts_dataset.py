from torch.utils.data import Dataset
from tqdm import tqdm

class CustomPromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        model_type='phi',
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.model_type = model_type

        # chat_template
        input_key = getattr(self.strategy.args, "input_key", None)
        
        self.prompts = []
        self.prompts = []
        print_one = True
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            messages = data[input_key][:-1]

            if 'phi' in self.model_type.lower():
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).replace(self.tokenizer.eos_token, '')
            else:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if print_one:
                print('\nprompt:', prompt)
                print_one = False
            self.prompts.append({"prompt": prompt, "messages": messages, "original_output": data["chosen"][-1]["content"].strip()})
        
    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
