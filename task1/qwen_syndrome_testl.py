from transformers import Qwen2ForSequenceClassification, Qwen2Tokenizer
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import json, random
import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from peft import PeftModel

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用CUDA

# 加载模型和分词器
model_name = "/mnt/qwen2.5-1.5b-instruct/"
# 创建标签到索引的映射
label_to_id = {
    "气虚血瘀证": 0,
    "痰瘀互结证": 1,
    "气阴两虚证": 2,
    "气滞血瘀证": 3,
    "肝阳上亢证": 4,
    "阴虚阳亢证": 5,
    "痰热蕴结证": 6,
    "痰湿痹阻证": 7,
    "阳虚水停证": 8,
    "肝肾阴虚证": 9,
}
# 创建 id_to_label 字典
id_to_label = {v: k for k, v in label_to_id.items()}

num_labels = len(label_to_id)  # 根据你的标签数量设置num_labels

# 加载预训练的 Qwen2 模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

class MultiLabelModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(MultiLabelModel, self).__init__()
        self.qwen = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.qwen.config.pad_token_id = 151643  # 定义pad token，模型才会忽略后面那些pad而是把真正最后一个token的hidden state用于分类
        self.loss_fn = nn.BCEWithLogitsLoss()  # 多标签分类损失函数

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
model = MultiLabelModel(model_name, num_labels=num_labels)
#合并lora
model = PeftModel.from_pretrained(model, model_id='/home/zhaozhizhuo22/CCL/save_model_1.5b_syndrome/')
for parameter in model.parameters():
    parameter.requires_grad = False

# 将模型分布到多个GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[2, 4])  # 在 GPU 卡 0 和 卡 2 上分布
model.to(device)

# 准备输入文本
texts = []
true_label = []
ID = []
prompt = '下面是一位患者的症状和中医望闻切诊，你需要依靠这些信息判断患者的证型，这类证型可能是一个也可能是两个：'
with open('./dataset/TCM-TBOSD-test-B.json','r',encoding='utf-8') as file:
    data = json.load(file)
    for line in data:
        t = prompt + line['症状'] + line['中医望闻切诊']
        texts.append(t)
        labels = [0] * num_labels  # 初始化多热编码向量（全 0）
        label_num = [label_to_id[i] for i in line['证型'].split('|')]
        for i in label_num:  # 遍历每个标签
            labels[i] = 1  # 将对应位置设置为 1
        true_label.append(labels)
        ID.append(line['ID'])


# 对文本进行编码
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs['logits']
predictions = torch.sigmoid(logits)
# threshold = find_threshold_micro(logits.cpu().detach().numpy(), torch.tensor(true_label).cpu().detach().numpy())
# for i in predictions:
#     print(i)
# 使用 torch.topk 获取每行的前两大值
topk_values, topk_indices = torch.topk(predictions, 2, dim=1)
# 获取最大值和次大值
top_max_values = topk_values[:, 0]  # 第一大值
second_max_values = topk_values[:, 1]  # 第二大值
# 对于每个 batch，检查最大值和次大值的差值
diff = top_max_values - second_max_values
# 如果差值大于0.2，则将次大值设置为最大值
second_max_values[diff > 0.1] = top_max_values[diff > 0.1]
predictions = torch.eq(predictions, second_max_values.unsqueeze(1))

# 检查对应位置是否相同
# 使用 np.all 逐行比较证型的正确率
yhat_raw_syndrome = predictions.to(torch.int).cpu().detach().numpy()
y_syndrome = torch.tensor(true_label).cpu().detach().numpy()
comparison_syndrome = np.all(yhat_raw_syndrome == y_syndrome, axis=1)
matching_rows_count_syndrome = np.sum(comparison_syndrome)
ACC_syndrome = matching_rows_count_syndrome / yhat_raw_syndrome.shape[0]
print('ACC:{}'.format(ACC_syndrome))

# 写入json文件中
results = []
yhat_raw_syndrome = yhat_raw_syndrome.tolist()
for num, i in enumerate(yhat_raw_syndrome):
    result = {}
    # 使用 enumerate 查找值为 1 的位置
    positions = [index for index, value in enumerate(i) if value == 1]
    # 将位置转换为标签
    labels = [id_to_label[position] for position in positions]
    result['ID'] = ID[num]
    result['证型'] = '|'.join(labels)
    results.append(result)
with open('/home/zhaozhizhuo22/CCL_baseline/output/syndrome.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
