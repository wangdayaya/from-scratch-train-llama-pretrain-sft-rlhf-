import torch
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoTokenizer, TrainingArguments, Trainer, DefaultDataCollator

from data.dataset import VLMDataset
from vllm_from_llama.model.vlm_config import VLMConfig
from vllm_from_llama.model.vlm import VLM, VLMCallback


def main():

    output_dir = "vlm_sft_hq_0402"
    # 加载配置
    print("加载 config")
    config = VLMConfig(max_seq_len=1024)

    # 加载分词器
    print("加载 tokenzier")
    tokenizer = AutoTokenizer.from_pretrained(r"D:\minimind\model\minimind_tokenizer")
    config.vocab_size = len(tokenizer)
    print(config)

    # 加载纯语言模型权重
    model = VLM(config)
    _, preprocess = VLM.get_vision_model()
    state_dict = torch.load(r'D:\PycharmProjects\from scratch train llama\vllm_from_llama\final_model\vlm_pretrain_hq_0401_state_dict.pth')
    model.load_state_dict(state_dict, strict=False)

    # 冻结 vision_encoder 的所有参数
    for name, param in model.named_parameters():
        if 'vision_encoder' in name:
            param.requires_grad = False

    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    trained_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    print(f'VLM总参数量：{num_params} B')  # 0.278B
    print(f'VLM可训练参数量：{trained_num_params} B')  # VLM可训练参数量：0.128615424 B ,其中投影层 393728 ，LM  0.128221696B ，

    # 数据
    print("Start Load Train Data...")
    train_ds = VLMDataset(r"D:\minimind-v_dataset\sft_vlm_data.jsonl",
                           r"D:\minimind-v_dataset\sft_images",
                          tokenizer,
                          preprocess=preprocess,
                          image_special_token=config.image_special_token,
                          max_seq_len=config.max_seq_len)
    print(f"训练数据{len(train_ds)}")

    # trainer
    print("Start set TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        seed=42,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        num_train_epochs=3,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        logging_steps=10,
        bf16=True,
        # dataloader_num_workers=4,
        # deepspeed=config.deepspeed
    )

    print("初始化 swanlab ")
    swanlab_callback = SwanLabCallback(
        project="vlm_from_scratch_sft",
        experiment_name=output_dir,
        config={**config.__dict__, 'dataset': 'hq', 'train_data_num':len(train_ds),  "参数量":  f"{num_params}B", "可训练参数量":  f"{trained_num_params}B"}
    )
    print("Start set Trainer............")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
        callbacks=[
            swanlab_callback,
            VLMCallback(tokenizer, preprocess, config, generate_every=50, max_new_length=50),
        ]
    )
    print("Start Training...............")
    trainer.train(resume_from_checkpoint=False)
    torch.save(model.state_dict(), f"./final_model/{output_dir}_state_dict.pth")
    torch.save(model, f"./final_model/{output_dir}.pth")


if __name__ == '__main__':
    main()
