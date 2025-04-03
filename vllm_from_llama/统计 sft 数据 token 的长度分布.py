import json

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

print("加载 tokenzier")
tokenizer = AutoTokenizer.from_pretrained(r"D:\minimind\model\minimind_tokenizer")


def _create_chat_prompt(conversations):
    messages = []
    for i, turn in enumerate(conversations['conversations']):
        role = 'user' if i % 2 == 0 else 'assistant'
        messages.append({"role": role, "content": turn['content'].replace('<image>', '@' * 196)})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


L = []
token_L = []
cnt = 0
with open(r"D:\minimind-v_dataset\sft_vlm_data.jsonl", "r", encoding="utf-8") as rfd:
    for line in tqdm(rfd.readlines()):
        data = json.loads(line)
        total = 0
        for c in data['conversations']:
            total += len(c['content'])
        L.append(total + 196 + 50)

        prompt = _create_chat_prompt(data)
        input_ids = tokenizer(prompt, )['input_ids']
        token_L.append(len(input_ids))

print(np.percentile(L, [10, 30, 50, 70, 90, 95, 99, 99.999]))
print(np.percentile(token_L, [10, 30, 50, 70, 90, 95, 99, 99.999]))

