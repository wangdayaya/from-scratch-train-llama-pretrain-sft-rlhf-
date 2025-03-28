# 初始化分词器
from concurrent.futures.thread import ThreadPoolExecutor

from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("D:\Qwen2.5-3B-Instruct")
file_paths = [r"D:\PycharmProjects\from scratch train llm\data\all_train.txt",
              r"D:\PycharmProjects\from scratch train llm\data\all_val.txt"]
MAX_LINES_PER_THREAD = 100000


def process_lines(lines):
    tokens_count = 0
    for line in lines:
        tokens = tokenizer.tokenize(line)
        tokens_count += len(tokens)
    return tokens_count


# 主函数
def main():
    all_lines = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            all_lines.extend(file.readlines())
    batches = [all_lines[i:i + MAX_LINES_PER_THREAD] for i in range(0, len(all_lines), MAX_LINES_PER_THREAD)]
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_lines, batches), total=len(batches), desc="Processing batches"))
    total_tokens = sum(results)
    print(f"Total number of tokens: {total_tokens}")

main()  # Total number of tokens: 2508051094
