import json
from torch.utils.data import Dataset
import numpy as np
import torch


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        mx, mn, total = 0, 100000, 0
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                for text in data['text'].split("</s> <s>"):
                    t = self.tokenizer.bos_token + text.strip().replace("</s>", "").replace("<s>",
                                                                                            "") + self.tokenizer.eos_token
                    samples.append(t)
                    mx = max(mx, len(t))
                    mn = min(mn, len(t))
                    total += len(t)
        for line in samples[:1] + samples[10000:10001] + samples[1000000:1000001]:
            print(line)
        print("mx:", mx, "mn:", mn, "mean:", total / len(samples))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = self.samples[index]
        encoding = self.tokenizer(text, max_length=self.max_seq_len, padding="max_length", truncation=True)
        input_ids = encoding.input_ids
        X = np.array(input_ids[:-1]).astype(np.int64)
        Y = np.array(input_ids[1:]).astype(np.int64)
        attention_mask = np.array(encoding.attention_mask[1:]).astype(np.int64)
        return {
            'input_ids': torch.from_numpy(X),
            'labels': torch.from_numpy(Y),
            'attention_mask': torch.from_numpy(attention_mask),
        }


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_seq_len)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_seq_len]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        attention_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return {
            "input_ids": X,
            "labels": Y,
            "attention_mask": attention_mask
        }


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)

                chosen = 0
                chosen += len(obj['chosen'][0]['content'])
                chosen += len(obj['chosen'][1]['content'])

                rejected = 0
                rejected += len(obj['rejected'][0]['content'])
                rejected += len(obj['rejected'][1]['content'])
                if chosen > max_seq_len or rejected > max_seq_len:
                    continue

                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上

        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)

        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_seq_len, padding='max_length')
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_seq_len, padding='max_length')

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)

        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return [x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected]

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_seq_len)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class DPODataCollator:
    def __call__(self, features):
        input_ids = []
        labels = []
        attention_mask = []

        for feature in features:
            input_ids.append(feature[0].tolist())
            labels.append(feature[1].tolist())
            attention_mask.append(feature[2].tolist())
        for feature in features:
            input_ids.append(feature[3].tolist())
            labels.append(feature[4].tolist())
            attention_mask.append(feature[5].tolist())

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int64)
        }
