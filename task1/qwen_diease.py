from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import json, random
from datasets import Dataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2" 

# 一些参数
model_name = "/mnt/qwen2.5-0.5-instruct/"  # 模型名或者本地路径
file_path = '/home/zhaozhizhuo22/CCL/llm_dataset/train.jsonl'  # 定义训练集路径
save_path = "/home/zhaozhizhuo22/CCL/save_model_7b/"

num_train_epochs=10
per_device_train_batch_size=2
per_device_eval_batch_size=2
warmup_steps=50
weight_decay=0.01
logging_steps=1
use_cpu=False

# 创建标签到索引的映射
label_to_id = {
    "胸痹心痛病": 0,
    "心衰病": 1,
    "眩晕病": 2,
    "心悸病": 3
}

num_labels = len(label_to_id)  # 根据你的标签数量设置num_labels

# 加载预训练的 Qwen2 模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)  
print(model)
model.config.pad_token_id = 151643  # 定义pad token，模型才会忽略后面那些pad而是把真正最后一个token的hidden state用于分类

# 读取jsonl文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

data = read_jsonl(file_path)
random.shuffle(data)

# 将文本标签转换为数值标签
for example in data:
    example['label'] = label_to_id[example['label']]

# 检查标签范围
for example in data:
    assert 0 <= example['label'] < len(label_to_id), f"Label out of range: {example['label']}"    

# 将数据转换为datasets库的Dataset对象
dataset = Dataset.from_list(data)

# 将数据集拆分为训练集和验证集
dataset = dataset.train_test_split(test_size=0.2)

# 定义一个函数来处理数据集中的文本
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, return_tensors="pt")

# 对数据集进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir=save_path,                           # 输出目录
    num_train_epochs=num_train_epochs,              # 训练的epoch数
    per_device_train_batch_size=per_device_train_batch_size,    # 每个设备的训练batch size
    per_device_eval_batch_size=per_device_eval_batch_size,      # 每个设备的评估batch size
    warmup_steps=warmup_steps,                  # 预热步数
    weight_decay=weight_decay,                  # 权重衰减
    logging_dir=save_path,                      # 日志目录
    logging_steps=logging_steps,
    evaluation_strategy="epoch",
    save_strategy="epoch",    # 每个epoch保存一次检查点
    save_total_limit=3,       # 最多保存3个检查点，旧的会被删除
    use_cpu=False
)

# 定义Trainer
trainer = Trainer(
    model=model,                                    # 模型
    args=training_args,                             # 训练参数
    train_dataset=encoded_dataset['train'],         # 训练数据集
    eval_dataset=encoded_dataset['test']            # 评估数据集
)

# 打印训练集和验证集中的一些样本
print("Train dataset sample:")
print(encoded_dataset['train'][0])  # 打印训练集中的第一个样本

print("Eval dataset sample:")
print(encoded_dataset['test'][0])  # 打印验证集中的第一个样本

# 开始训练
trainer.train()
trainer.save_state()
trainer.save_model(output_dir=save_path)
tokenizer.save_pretrained(save_path)