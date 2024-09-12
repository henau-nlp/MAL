import random

def split_dataset(data_lines, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    total_sentences = len(data_lines)
    indices = list(range(total_sentences))
    random.shuffle(indices)

    train_split = int(train_ratio * total_sentences)
    val_split = int((train_ratio + val_ratio) * total_sentences)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    train_data = [data_lines[i] for i in train_indices]
    val_data = [data_lines[i] for i in val_indices]
    test_data = [data_lines[i] for i in test_indices]

    return train_data, val_data, test_data

# 读取数据文件
with open('alldata.txt', 'r', encoding='utf-8') as file:
    data_lines = file.read().split('\n\n')  # 使用空行分割句子

# 分割数据集
train_data, val_data, test_data = split_dataset(data_lines)

# 写入训练集文件
with open('6train.txt', 'w', encoding='utf-8') as file:
    file.write('\n\n'.join(train_data))

# 写入验证集文件
with open('6dev.txt', 'w', encoding='utf-8') as file:
    file.write('\n\n'.join(val_data))

# 写入测试集文件
with open('6test.txt', 'w', encoding='utf-8') as file:
    file.write('\n\n'.join(test_data))

print("数据集已成功分割为训练集、验证集和测试集。")
