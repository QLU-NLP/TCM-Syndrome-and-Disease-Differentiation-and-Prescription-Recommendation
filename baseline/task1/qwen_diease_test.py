from transformers import Qwen2ForSequenceClassification, Qwen2Tokenizer
import torch
import json

# 加载模型和分词器
model_name = "/home/zhaozhizhuo22/CCL_baseline/save_model_0.5b_diease/"
tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
model = Qwen2ForSequenceClassification.from_pretrained(model_name).to('cuda:2')
for parameter in model.parameters():
    parameter.requires_grad = False

# 创建标签到索引的映射
label_to_id = {
    "胸痹心痛病": 0,
    "心衰病": 1,
    "眩晕病": 2,
    "心悸病": 3
}
# 创建 id_to_label 字典
id_to_label = {v: k for k, v in label_to_id.items()}

# 准备输入文本
texts = []
true_label = []
ID = []
with open('./dataset/TCM-TBOSD-test-B.json','r',encoding='utf-8') as file:
    data = json.load(file)
    for line in data:
        t = line['症状'] + line['中医望闻切诊']
        texts.append(t)
        true_label.append(label_to_id[line['疾病']])
        ID.append(line['ID'])


# 对文本进行编码
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to('cuda:2')

# 进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)

# 检查对应位置是否相同
equal_positions = (predictions == torch.tensor(true_label).to('cuda:2'))
# 统计相同的位置数量
num_equal = equal_positions.sum().item()

acc = num_equal / len(true_label)
print('ACC:{}'.format(acc))

results = []
for num, i in enumerate(predictions):
    results.append({"ID": ID[num], "疾病": id_to_label[i.item()]})

with open('/home/zhaozhizhuo22/CCL_baseline/output/diease.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)