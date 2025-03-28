import json
import os
import random
from functools import reduce

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


# -----------------  找长度大约为 1000 的字符串进行 tokenizer ，查看分词后的 token 数量是否在1024以内------------------------
def cal_token_from():
    tokenizer = AutoTokenizer.from_pretrained("D:\Qwen2.5-3B-Instruct")
    target = "我海军导弹驱逐舰西安舰主炮射击瞬间。解放军报记者&nbsp;罗朝文&nbsp;摄 我海军导弹驱逐舰西安舰主炮射击瞬间。解放军报记者罗朝文摄海风呼啸，白浪翻腾。按照“环太平洋-2016”演习计划，当地时间7月21日上午，一场主炮对海上目标射击演练，在夏威夷西南约200海里的太平洋上举行。上午8时许，由中国、美国、法国、印尼4国海军组成的175特混编队展开战术机动。美国海岸警卫队“斯特拉顿”号炮舰、法国海军“牧月”号护卫舰、印尼海军“迪波内格罗”号护卫舰、中国海军导弹护卫舰衡水舰、导弹驱逐舰西安舰、综合补给舰高邮湖舰以及和平方舟医院船间距1海里，依次组成单纵队，向射击海域机动。9时许，由“斯特拉顿”号炮舰提前布设的浮体靶出现在编队右侧约6海里海域。这个被美国海军称为“番茄杀手”的浮体靶是一个直径约5米的橘色球形充气靶，专用于火炮对海射击。相比于此前进行的主炮对海上经纬度点射击演练，此次射击实体目标，更便于直观判断参演各舰的射击精度。根据气象观测，演习区域数百海里外，一个热带风暴正向演练海区靠近，风浪与涌浪叠加，舰体摇摆较前几日明显加大。从西安舰右舷遥望，橘色的“番茄杀手”更是飘忽不定、若隐若现。中国舰艇编队参谋长陈德楠告诉记者，这次投放的靶标，雷达反射面非常小，受洋流、风速等因素影响，火炮命中数海里外的靶标并非易事，尤其是在今天风浪较大的海况下，更是对射击精度的考验。9时30分，射击演练正式开始。在法国海军“牧月”号护卫舰、印尼海军“迪波内格罗”号护卫舰射击后，中国海军衡水舰登场亮相。远方洋面上，“番茄杀手”依然摇摆在波峰浪谷间。衡水舰上，硝烟乍起，主炮开始射击，16发炮弹相继飞向目标，“番茄杀手”边溅起道道水柱，弹着密集覆盖番茄靶，将橘红色的浮体靶标抛起、打瘪。西安舰内，战斗警报骤然拉响。测定靶标距离、射击方位、锁定靶标……西安舰各战位密切协同，迅速做好射击准备。主炮注意，打击目标0001批!”“快放4发，放!”一声令下，西安舰主炮腾起阵阵硝烟，炮弹呼啸而出，几秒钟后，弹丸在目标水域掀起冲天水柱，“番茄杀手”被彻底摧毁，湛蓝的洋面上，只留下翻腾的浪花。登上西安舰的美军观察员梅多勒斯在射击完成后不禁对中国海军表示赞赏。他说：“中国军舰的表现非常专业，精准命中了\"番茄杀手\"，我能够参与这次演练感觉非常棒！” 来源：国防部网\n"
    tokens = tokenizer.tokenize(target)
    token_count = len(tokens)
    print(f"Token数量: {token_count}")


# -----------------  计算多个文件数据的总token数量------------------------
def sum_of_token_from_files():
    tokenizer = AutoTokenizer.from_pretrained("D:\Qwen2.5-3B-Instruct")
    total_tokens = 0
    for file_path in [r"D:\PycharmProjects\from scratch train llm\data\all_train.txt", r"D:\PycharmProjects\from scratch train llm\data\all_val.txt"]:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in tqdm(file.readlines()):
                tokens = tokenizer.tokenize(line)
                total_tokens += len(tokens)
    return total_tokens


# -----------------将文本按照 1024 的长度进行截断，并且保证句子的通顺 -----------------
def split_and_reassemble(text, max_length=512, delimiter="。"):
    sentences = text.split(delimiter)
    sentences = [s.strip() for s in sentences if s.strip()]
    result = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip("“'”“") + delimiter
        current_chunk += sentence
        if  max_length > len(current_chunk) >= max_length:
            result.append(current_chunk)
            current_chunk = ""
    if current_chunk and max_length >len(current_chunk)>50:
        result.append(current_chunk)
    return set(result)


# ----------------- 处理 news 数据------------------------
def handle_new():
    rows = set()
    output_file_path = r"D:\PycharmProjects\from scratch train llm\data\news.txt"
    for file in [r"D:\PycharmProjects\from scratch train llm\data\news2016zh_valid.json",
                 r"D:\PycharmProjects\from scratch train llm\data\news2016zh_train.json"]:
        print(f"正在处理{file}")
        with open(file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for i, line in tqdm(enumerate(lines)):
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if "content" in data:
                        rows |= split_and_reassemble(data["content"])
    print(f"一共{len(rows)}条数据")
    lengths = list(map(len, rows))
    max_length = max(lengths)
    min_length = min(lengths)
    average_length = reduce(lambda x, y: x + y, lengths) / len(lengths)
    octiles = np.percentile(lengths, [95, 99])
    print(max_length, min_length, average_length, octiles)
    with open(output_file_path, "a", encoding="utf-8") as file:
        for text in rows:
            if text:
                file.write(text + "\n")



# ----------------- 处理 web 数据------------------------
def handle_web():
    rows = set()
    output_file_path = r"D:\PycharmProjects\from scratch train llm\data\webs.txt"
    for file in [r"D:\PycharmProjects\from scratch train llm\data\web_text_zh_testa.json",
                 r"D:\PycharmProjects\from scratch train llm\data\web_text_zh_train.json",
                 r"D:\PycharmProjects\from scratch train llm\data\web_text_zh_valid.json"]:
        print(f"正在处理{file}")
        with open(file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for i, line in tqdm(enumerate(lines)):
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if "content" in data:
                        rows |= split_and_reassemble(data["content"])
    print(f"一共{len(rows)}条数据")
    lengths = list(map(len, rows))
    max_length = max(lengths)
    min_length = min(lengths)
    average_length = reduce(lambda x, y: x + y, lengths) / len(lengths)
    octiles = np.percentile(lengths, [95, 99])
    print(max_length, min_length, average_length, octiles)  # 511 51 178.39043753409345 [224. 269.]
    with open(output_file_path, "a", encoding="utf-8") as file:
        for text in rows:
            if text:
                file.write(text + "\n")




# ----------------- 处理 wiki 数据------------------------

def handle_wiki():
    directory_path = r"D:\PycharmProjects\from scratch train llm\data\wiki_zh"
    output_file_path = r"D:\PycharmProjects\from scratch train llm\data\wikis.txt"

    def find_all_files(directory):
        all_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        return all_files

    files = find_all_files(directory_path)

    rows = set()
    for file in tqdm(files):
        with open(file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if "text" in data:
                        rows |= split_and_reassemble(data["text"])
    print(f"一共{len(rows)}条数据")
    lengths = list(map(len, rows))
    max_length = max(lengths)
    min_length = min(lengths)
    average_length = reduce(lambda x, y: x + y, lengths) / len(lengths)
    octiles = np.percentile(lengths, [95, 99])
    print(max_length, min_length, average_length, octiles)
    with open(output_file_path, "a", encoding="utf-8") as file:
        for text in rows:
            if text:
                file.write(text + "\n")



# ----------------- 预览最后 5 条数据------------------------
def preview_last_n_lines(file_path, n=5):
    """
    预览文件的最后 n 行。
    :param file_path: 文件路径
    :param n: 需要预览的行数，默认为 5
    """
    with open(file_path, "r", encoding="utf-8") as file:
        # 获取文件的总行数
        total_lines = sum(1 for _ in file)

        # 如果文件行数少于 n，直接读取所有行
        if total_lines <= n:
            file.seek(0)  # 重置文件指针到开头
            last_lines = file.readlines()
        else:
            # 计算需要跳过的行数
            skip_lines = total_lines - n
            file.seek(0)  # 重置文件指针到开头
            for _ in range(skip_lines):
                file.readline()  # 跳过前面的行
            last_lines = file.readlines()
    return last_lines


# -----------------  将三个文件合并成一个 txt 文件------------------------
def merge_shuffle_and_split(file_paths, output_file_a, output_file_b, train_ratio=0.999, chunk_size=1024 * 1024 * 1024):
    """
    合并多个大文件，混洗数据，并将前 train_ratio 的行写入文件 A，剩余的行写入文件 B。
    """
    # 1. 逐块读取所有文件并合并
    all_lines = []
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                lines = chunk.splitlines()
                all_lines.extend(lines)

    random.shuffle(all_lines)

    total_lines = len(all_lines)
    split_index = int(total_lines * train_ratio)

    with open(output_file_a, "w", encoding="utf-8") as file_a, open(output_file_b, "w", encoding="utf-8") as file_b:
        file_a.writelines(line + "\n" for line in all_lines[:split_index])
        print(f"Train data written to {output_file_a}")

        file_b.writelines(line + "\n" for line in all_lines[split_index:])
        print(f"Eval data written to {output_file_b}")



# print(preview_last_n_lines(r"D:\PycharmProjects\from scratch train llm\data\all_train.txt"))
# print(preview_last_n_lines(r"D:\PycharmProjects\from scratch train llm\data\news.txt"))
# print(preview_last_n_lines(r"D:\PycharmProjects\from scratch train llm\data\webs.txt"))
# print(preview_last_n_lines(r"D:\PycharmProjects\from scratch train llm\data\wikis.txt"))
# ---------------------------

# handle_new()  # 一共572347条数据 511 51 313.0684287678628 [493. 508.]
# handle_web()  #  # 一共3576567条数据 511 51 178.39043753409345 [224. 269.]
# handle_wiki()   #  一共720506条数据 511 51 165.70916133939204 [418. 491.]

merge_shuffle_and_split([r"D:\PycharmProjects\from scratch train llm\data\news.txt",
                         # r"D:\PycharmProjects\from scratch train llm\data\webs.txt",
                         # r"D:\PycharmProjects\from scratch train llm\data\wikis.txt"
                         ],
                        r"D:\PycharmProjects\from scratch train llm\data\all_train.txt",
                        r"D:\PycharmProjects\from scratch train llm\data\all_val.txt",
                        )


# 读取文件
# file_path = r"D:\PycharmProjects\from scratch train llm\data\wikis.txt"
# with open(file_path, "r", encoding="utf-8") as file:
#     for line in file.readlines():
#         # 打印原始字符串
#         print("原始字符串内容：")
#         print(line)

# sum_of_token_from_files()