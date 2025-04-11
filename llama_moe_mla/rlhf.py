import torch
import torch.nn.functional as F
from swanlab.integration.transformers import SwanLabCallback
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, TrainingArguments, Trainer

from data.dataset import DPODataCollator, DPODataset
from llama_moe_mla.model.llama_moe_mla import ChatCallback, LlamaForCausalLM
from llama_moe_mla.model.llama_moe_mla_config import LMConfig


def logits_to_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def mask_logits(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels_masks shape: (batch_size, seq_len)
    new_logits = []
    for logit, label in zip(logits, labels):
        new_logits.append(logit[label != 0].sum().unsqueeze(0))

    return new_logits


def dpo_loss(ref_probs, probs, beta):
    def split_probs(probs):
        len_chosen = int(len(probs) // 2)
        chosen_data = probs[:len_chosen]
        reject_data = probs[len_chosen:]
        return torch.cat(chosen_data), torch.cat(reject_data)

    ref_chosen_probs, ref_reject_probs = split_probs(ref_probs)
    chosen_probs, reject_probs = split_probs(probs)
    pi_logratios = chosen_probs - reject_probs
    ref_logratios = ref_chosen_probs - ref_reject_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


class DPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        attention_mask = inputs['attention_mask']

        with torch.no_grad(), autocast():
            ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
        ref_probs = logits_to_probs(ref_logits, labels)
        ref_probs = mask_logits(ref_probs, labels)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        probs = logits_to_probs(logits, labels)
        probs = mask_logits(probs, labels)

        loss = dpo_loss(ref_probs, probs, 0.1)
        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
    # 加载配置
    print("加载 config")
    config = LMConfig(max_seq_len=1024)

    # 加载分词器
    print("加载 tokenzier")
    tokenizer = AutoTokenizer.from_pretrained(r"D:\minimind\model\minimind_tokenizer")
    config.vocab_size = len(tokenizer)
    print(config)

    # 加载模型
    print("加载训练好的模型")
    model = LlamaForCausalLM(config)
    model.load_state_dict(torch.load('./final_model/llama_moe_mla_sft_hq_0409_state_dict.pth'))
    ref_model = LlamaForCausalLM(config)
    ref_model.load_state_dict(torch.load('./final_model/llama_moe_mla_sft_hq_0409_state_dict.pth')).eval().to('cuda')

    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of model parameters: {num_params / 1000 ** 3}B")
    print(f"Total number of ref_model parameters: {num_params / 1000 ** 3}B")
    print(f"GPU 是否可用{torch.cuda.is_available()}，有{torch.cuda.device_count()}个GPU")

    # 数据
    print("Start Load Train and Validation Data...")
    train_set = DPODataset(r"D:\minimind_dataset\dpo_train.jsonl", tokenizer, config.max_seq_len)
    val_set = DPODataset(r"D:\minimind_dataset\dpo_val.jsonl", tokenizer, config.max_seq_len)
    print(f"训练数据{len(train_set)}，验证数据{len(val_set)}")

    # trainer
    print("Start set TrainingArguments...")
    output_dir = "llama_moe_mla_dpo_0410"
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        seed=42,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=2,
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
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=1,
        bf16=True,
        # deepspeed=config.deepspeed
    )

    print("初始化 swanlab ")
    swanlab_callback = SwanLabCallback(
        project="llama_moe_mla_from_scratch_dpo",
        experiment_name=output_dir,
        config={**config.__dict__, 'dataset': '搜集合并了网上的多份数据', 'train_data_num':len(train_set), 'val_data_num':len(val_set), "参数量":  f"{num_params / 1000 ** 3}B"}
    )
    print("Start set Trainer............")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=DPODataCollator(),
        callbacks=[
            swanlab_callback,
            ChatCallback(tokenizer, generate_every=50, max_new_length=1024)]
    )
    print("Start Training...............")
    trainer.train(resume_from_checkpoint=False)
    torch.save(model.state_dict(), f"./final_model/{output_dir}_state_dict.pth")
    torch.save(model, f"./final_model/{output_dir}.pth")

