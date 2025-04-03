# 数据说明
1. pretrain 使用了 minimind 项目中的 pretrain_hq.json
2. sft 使用了 minimind 项目中的 sft_mini_512.json
3. dpo 使用了自己从 huggingface 上面 搜集到的中文偏好数据，如下：
    
    ```
     DPO-En-Zh-20k 中的 1 万条中文数据集
     dpo-toxic-zh 中的 4750 条中文数据，这个数据很敏感，慎用！！
      zhihu_rlhf_3k 中的 3460 条中文数据
     rlhf_reward_single_round-chinese-zhtw 中的 19900 条中文数据
     总共 43067 条数据，都统一处理成一样的格式。
     经过数据统计发现几乎绝大部分的数据的 chosen 或者 rejected 的问答长度都小于 1024 ，所以将大于 1024 的都过滤掉，最后剩下 38896 条数据。
   ```
