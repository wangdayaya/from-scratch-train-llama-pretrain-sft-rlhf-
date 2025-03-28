import json

import numpy as np

mx_chosen = 0
mn_chosen = 1000000000
mx_rejected = 0
chosen_l = []
mn_rejected = 1000000000
total_chosen = 0
total_rejected = 0
rejected_l = []
cnt = 0
with open(r"D:\PycharmProjects\from scratch train llm\data\dpo.jsonl", "r", encoding="utf-8") as rfd:
    for line in rfd.readlines():
        data = json.loads(line)
        chosen = 0
        chosen += len(data['chosen'][0]['content'])
        chosen += len(data['chosen'][1]['content'])
        total_chosen += chosen

        rejected = 0
        rejected += len(data['rejected'][0]['content'])
        rejected += len(data['rejected'][1]['content'])
        total_rejected += rejected
        if chosen > 1024 or rejected > 1024:
            continue
        cnt += 1
        chosen_l.append(chosen)
        mx_chosen = max(mx_chosen, chosen)
        mn_chosen = min(mn_chosen, chosen)

        rejected_l.append(rejected)
        mx_rejected = max(mx_rejected, rejected)
        mn_rejected = min(mn_rejected, rejected)

print(cnt)
print("mx_chosen:", mx_chosen)
print("mn_chosen:", mn_chosen)
print("mean_chosen:", total_chosen / 43067)
print(np.percentile(chosen_l, [10, 30, 50, 70, 90, 95, 99, 99.999]))

print("mx_rejected:", mx_rejected)
print("mn_rejected:", mn_rejected)
print("mean_rejected:", total_rejected / 43067)
print(np.percentile(rejected_l, [10, 30, 50, 70, 90, 95, 99, 99.999]))
