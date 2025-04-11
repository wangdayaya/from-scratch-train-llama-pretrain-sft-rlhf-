import collections
import json
import os
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama.model.llama_config import LMConfig
import pandas as pd


def read_ceval_data():
    data_frames = []
    result = collections.defaultdict(dict)
    for root, dirs, files in os.walk('../ceval'):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                for row in df.to_dict(orient='records'):
                    row['sub'] = file.replace(".csv","")
                    if 'test' in row['sub']:
                        result[row['sub']][row['id']] = ""
                    data_frames.append(row)
    return data_frames, result


def get_answer_by_qwen(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
    generated_ids = qwen_model.generate(**model_inputs, max_new_tokens=10)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


if __name__ == '__main__':
    data, result = read_ceval_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qwen_model = AutoModelForCausalLM.from_pretrained("D:\Qwen2.5-7B-Instruct", torch_dtype="auto", device_map="auto")
    qwen_tokenizer = AutoTokenizer.from_pretrained("D:\Qwen2.5-7B-Instruct")

    config = LMConfig()
    tokenizer = AutoTokenizer.from_pretrained(r"D:\minimind\model\minimind_tokenizer")
    config.vocab_size = len(tokenizer)
    model = torch.load('final_models/sft_hq_0326.pth').to(device)

    total = len(data)
    not_legal = 0
    wrong = 0
    correct = 0
    for d in tqdm(data):
        prompt = f"""下列关于资本结构理论的说法中，不正确的是____。
    A、代理理论、权衡理论、有企业所得税条件下的MM理论，都认为企业价值与资本结构有关
    B、按照优序融资理论的观点，考虑信息不对称和逆向选择的影响，管理者偏好首选留存收益筹资，然后是发行新股筹资，最后是债务筹资
    C、权衡理论是对有企业所得税条件下的MM理论的扩展
    D、代理理论是对权衡理论的扩展
    答案是D
    
    某烟草公司2022年1月8日向烟农支付烟叶收购价款58万元，另支付了价外补贴10万元。该烟草公司1月收购烟叶应缴纳的烟叶税为____万元。
    A、11.6
    B、12.76
    C、13.6
    D、14.96
    答案是C
    
    根据公司法律制度的规定，公司董事的下列行为中，涉嫌违反勤勉义务的是____。
    A、擅自披露公司商业秘密
    B、将公司资金以个人名义开立账户存储
    C、无正当理由长期不出席董事会会议
    D、篡夺公司商业机会
    答案是C
    
    请认真理解下面的题意，并按照上面的格式选择出唯一正确的选项，不需要解释，直接给出 A、B、C、D 的其中一个即可。
    {d['question']}
    A、{d['A']}
    B、{d['B']}
    C、{d['C']}
    D、{d['D']}
    答案是"""
        messages = [{"role": 'user', "content": prompt}]
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = torch.tensor(tokenizer(new_prompt)['input_ids'], device=model.device).unsqueeze(0)

        start = input_ids.shape[1]
        with torch.no_grad(), autocast():
            outputs = model.generate(input_ids=input_ids, max_length=20, eos_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[0][start:], skip_special_tokens=True)

        answer = get_answer_by_qwen(f"文本：{decoded}\n上面的文本可能包含了选择题的答案 A、B、C、D 中的任意一个值，如果有请直接返回，不需要解释，如果没有则返回'没有答案'。")
        if 'val' in d['sub'] or 'dev' in d['sub']:
            if len(answer) == 1:
                if d['answer'] == answer:
                    correct += 1
                else:
                    wrong += 1
            else:
                not_legal += 1
            print(f"correct:{correct},wrong:{wrong},not_legal:{not_legal}")
        else:
            result[d['sub']][d['id']] = answer
    with open("eval_llama_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
