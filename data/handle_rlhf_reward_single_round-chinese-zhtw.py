import json

from datasets import load_dataset
from tqdm import tqdm


# 处理 dpo 数据
content = load_dataset("erhwenkuo/rlhf_reward_single_round-chinese-zhtw")

with open(r"D:\PycharmProjects\from scratch train llm\data\rlhf_reward_single_round-chinese-zhtw.jsonl", "w", encoding="utf-8") as wfd:
    for s in ["train", "test"]:
        for data in tqdm(content[s]):
                question = data['prompt']
                c = data['chosen']
                r = data['rejected']
                t = {
                    "chosen": [
                        {
                            "content": question,
                            "role": "user"
                        },
                        {
                            "content": c,
                            "role": "assistant"
                        }
                    ],
                    "rejected": [
                        {
                            "content": question,
                            "role": "user"
                        },
                        {
                            "content": r,
                            "role": "assistant"
                        }
                    ]
                }
                json_str = json.dumps(t, ensure_ascii=False)
                wfd.write(json_str + '\n')

