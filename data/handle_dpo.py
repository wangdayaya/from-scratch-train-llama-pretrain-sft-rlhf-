import json
import re

from tqdm import tqdm


# 处理 dpo 数据
with open(r"D:\PycharmProjects\from scratch train llm\data\dpo_zh.json", "r", encoding='utf-8') as rfd:
    content = rfd.read()
content = json.loads(content)

with open(r"D:\PycharmProjects\from scratch train llm\data\dpo.jsonl", "w", encoding="utf-8") as wfd:
    for data in tqdm(content):
            if len(data['conversations']) == 1:
                question = data['conversations'][0]['value']
            else:
                question = data['conversations'][-1]['value']
            c = data['chosen']['value']
            r = data['rejected']['value']
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

