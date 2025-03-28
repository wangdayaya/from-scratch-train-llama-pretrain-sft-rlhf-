from transformers import PreTrainedTokenizerFast

# 加载分词器
tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe-tokenizer.json")

# 测试分词器
text = "我们是中国人"
encoded = tokenizer.encode(text, return_tensors="pt")
print(encoded)

decoded = tokenizer.decode(encoded[0], skip_special_tokens=True)
print(decoded)