import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer

from model.llama_config import LMConfig
from model.llama import LlamaForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU



# pretrain
print("加载 config")
config = LMConfig()

print("加载 tokenizer")
tokenizer = AutoTokenizer.from_pretrained(r"D:\minimind\model\minimind_tokenizer")
config.vocab_size = len(tokenizer)
model = LlamaForCausalLM(config)
model.load_state_dict(torch.load('final_models/pretrain_hq_0325_state_dict.pth'))
# model = torch.load('final_models/pretrain_hq_0325.pth')
model.to(device)
model.eval()  # 设置为评估模式
input_text = tokenizer.bos_token + "你知道物理吗？"
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)  # 将输入数据移动到 GPU
model = LlamaForCausalLM(config)
model.load_state_dict(torch.load(r'/llama/final_models\pretrain_hq_0325_state_dict.pth'), )
model.to(device)
model.eval()  # 设置为评估模式
for s in ["给我一些写作的建议", "写一首关于小狗的诗歌", "挑选一些国内的旅游景点","给我一些健身的建议", "如何提高学习成绩"]:
    input_text = tokenizer.bos_token + s
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)  # 将输入数据移动到 GPU
    start = input_ids.shape[1]
    # 生成文本
    with torch.no_grad():  # 关闭梯度计算，混合精度加速 , autocast()
        outputs = model.generate(input_ids=input_ids, max_new_length=500, eos_token_id=tokenizer.eos_token_id)

    # 解码生成的文本
    decoded = tokenizer.decode(outputs[0][start:], skip_special_tokens=True)
    print(decoded)
    print("----------")


# sft
# print("加载 config")
# config = LMConfig()
# print("加载 tokenizer")
# tokenizer = AutoTokenizer.from_pretrained(r"D:\minimind\model\minimind_tokenizer")
# config.vocab_size = len(tokenizer)
#
# model = LlamaForCausalLM(config)
# model.load_state_dict(torch.load(r'D:\PycharmProjects\from scratch train llama\llama\final_models\sft_hq_0326_state_dict.pth'), )
# model.to(device)
# model.eval()  # 设置为评估模式
# for s in ["给我一些写作的建议", "写一首关于小狗的诗歌", "挑选一些国内的旅游景点","给我一些健身的建议", "如何提高学习成绩"]:
#     messages = [{"role": 'user', "content": s}]
#     new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#     input_ids = torch.tensor(tokenizer(new_prompt)['input_ids'], device=model.device).unsqueeze(0)
#     start = input_ids.shape[1]
#     # 生成文本
#     with torch.no_grad():  # 关闭梯度计算，混合精度加速   ,  autocast(),
#         outputs = model.generate(input_ids=input_ids, max_new_length=500, eos_token_id=tokenizer.eos_token_id)
#
#     # 解码生成的文本
#     decoded = tokenizer.decode(outputs[0][start:], skip_special_tokens=True)
#     print(decoded)
#     print("----------")

# rlhf
# model = LlamaForCausalLM(config)
# model.load_state_dict(torch.load('final_models/dpo_0326_state_dict.pth'), )
# # model = torch.load('final_models/dpo_0326.pth')
# model.to(device)
# model.eval()  # 设置为评估模式
# messages = [{"role": 'user', "content": "你知道如何炒股吗"}]
# new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
# input_ids = torch.tensor(tokenizer(new_prompt)['input_ids'], device=model.device).unsqueeze(0)


# start = input_ids.shape[1]
# # 生成文本
# with torch.no_grad(), autocast():  # 关闭梯度计算，混合精度加速
#     outputs = model.generate(input_ids=input_ids, max_new_length=600, eos_token_id=tokenizer.eos_token_id)
#
# # 解码生成的文本
# decoded = tokenizer.decode(outputs[0][start:], skip_special_tokens=True)
# print(decoded)
