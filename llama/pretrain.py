import torch
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoTokenizer, TrainingArguments, Trainer, DefaultDataCollator
from data.dataset import PretrainDataset
from llama.model.llama_config import LMConfig
from llama.model.llama import LlamaForCausalLM, GenerateTextCallback


def main():

    output_dir = "llama_pretrain_hq_0325"

    # 加载配置
    print("加载 config")
    config = LMConfig(flash_attention=True)

    # 加载分词器
    print("加载 tokenzier")
    tokenizer = AutoTokenizer.from_pretrained(r"D:\minimind\model\minimind_tokenizer")
    config.vocab_size = len(tokenizer)
    print(config)

    print("初始化 others")
    model = LlamaForCausalLM(config)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params / 1000 ** 3}B")
    print(f"GPU 是否可用{torch.cuda.is_available()}，有{torch.cuda.device_count()}个GPU")

    # 数据
    print("Start Load Train and Validation Data...")
    train_set = PretrainDataset(r"D:\minimind_dataset\pretrain_hq_train.jsonl", tokenizer, config.max_seq_len)
    val_set = PretrainDataset(r"D:\minimind_dataset\pretrain_hq_val.jsonl", tokenizer, config.max_seq_len)
    print(f"训练数据{len(train_set)}，验证数据{len(val_set)}")

    # trainer
    print("Start set TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        seed=42,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=10,
        gradient_checkpointing=False,
        num_train_epochs=2,
        learning_rate=5e-4,
        warmup_ratio=0.03,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=5000,
        logging_steps=10,
        bf16=True,
        # deepspeed=config.deepspeed
    )

    print("初始化 swanlab ")
    # swanlab_callback = SwanLabCallback(
    #     project="llama_from_scratch_pretrain",
    #     experiment_name=output_dir,
    #     config={**config.__dict__, 'dataset': 'hq', 'train_data_num':len(train_set), 'val_data_num':len(val_set), "参数量":  f"{num_params / 1000 ** 3}B"}
    # )
    print("Start set Trainer............")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
        callbacks=[
            # swanlab_callback,
            GenerateTextCallback(tokenizer, generate_every=100, max_new_length=100),
        ]
    )
    print("Start Training...............")
    trainer.train(resume_from_checkpoint=False)
    torch.save(model.state_dict(), f"llama/llama_final_models/{output_dir}_state_dict.pth")
    torch.save(model, f"llama/llama_final_models/{output_dir}.pth")


if __name__ == '__main__':
    main()
