import os

import torch
from PIL import Image
from torch.cuda.amp import autocast
from transformers import AutoTokenizer

from vllm_from_llama.model.vlm import VLM
from vllm_from_llama.model.vlm_config import VLMConfig

if __name__ == "__main__":
    device = torch.device("cuda")

    # 加载配置
    print("加载 config")
    config = VLMConfig(max_seq_len=1024)

    # 加载分词器
    print("加载 tokenzier")
    tokenizer = AutoTokenizer.from_pretrained(r"D:\minimind\model\minimind_tokenizer")
    config.vocab_size = len(tokenizer)
    print(config)

    model = VLM(config)
    state_dict = torch.load("final_model/vlm_sft_hq_0402_state_dict.pth")
    model.load_state_dict(state_dict, strict=False)

    vision_model, preprocess = VLM.get_vision_model()

    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    trained_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    print(f'VLM总参数量：{num_params} B')
    print(f'VLM可训练参数量：{trained_num_params} B')

    model.eval().to(device)
    vision_model.eval().to(device)


    def chat_with_vlm(prompt, pixel_tensors, image_names):
        messages = [{"role": "user", "content": prompt}]

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-config.max_seq_len + 1:]

        print(f'[Image]: {image_names}')
        with torch.no_grad(), autocast():
            e = tokenizer(new_prompt)
            input_ids = torch.tensor(e['input_ids'], device=device).unsqueeze(0)  # [1, 238]
            attention_mask = torch.tensor(e['attention_mask'], device=device).unsqueeze(0)  # [1, 238]
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                max_new_length=200,
                max_new_tokens=config.max_seq_len,
                pixel_tensors=pixel_tensors
            )
            print('🤖️: ', end='')
            print(tokenizer.decode(outputs.squeeze()[input_ids.shape[1]:].tolist(), skip_special_tokens=True), end='')
            print('\n')


    # 单图推理：每1个图像单独推理
    # image_dir = './eval_image'
    # prompt = f"{model.params.image_special_token}\n描述一下这个图像的内容。"
    # for image_file in os.listdir(image_dir):
    #     image = Image.open(os.path.join(image_dir, image_file)).convert('RGB')
    #     pixel_tensors = VLM.image2tensor(image, preprocess).to(device).unsqueeze(0)
    #     chat_with_vlm(prompt, pixel_tensors, image_file)



    # 多图推理：2个图像一起推理
    image_dir = './eval_multi_images/'
    prompt = (f"{config.image_special_token}\n"
              f"{config.image_special_token}\n"
              f"比较一下两张图像种小鸟的颜色是什么")
    pixel_tensors_multi = []
    for image_file in os.listdir(image_dir):
        image = Image.open(os.path.join(image_dir, image_file)).convert('RGB')
        pixel_tensors_multi.append(VLM.image2tensor(image, preprocess))
    pixel_tensors = torch.cat(pixel_tensors_multi, dim=0).to(device).unsqueeze(0)
    # 同样内容重复10次
    for _ in range(3):
        chat_with_vlm(prompt, pixel_tensors, (', '.join(os.listdir(image_dir))))
