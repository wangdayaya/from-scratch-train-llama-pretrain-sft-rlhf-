from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm


def get_training_corpus():
    dataset = []
    path = r"D:\PycharmProjects\from scratch train llm\data\all_val.txt"
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dataset.append(line)
    for i in tqdm(range(0, len(dataset), 1000)):
        yield dataset[i: i + 1000]

tokenizer = Tokenizer(models.BPE())
special_tokens = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
  ]
trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=special_tokens)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
tokenizer.save("bpe-tokenizer.json")
# 内存泄漏