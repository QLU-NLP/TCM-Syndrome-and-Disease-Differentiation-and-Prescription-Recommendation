import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import shutil
import json
import sys
import random
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import time
from transformers import get_linear_schedule_with_warmup
import numpy as np
import warnings
import torch.nn.functional as F
import math
# from peft import LoraModel, LoraConfig
from transformers import BertTokenizer
from transformers import BertConfig, BertModel, AdamW
# from sklearn.metrics import accuracy_score
from data_utils import TCM_SD_Data_Loader

# 设定随机种子值，以确保输出是确定的
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def find_threshold_micro(dev_yhat_raw, dev_y):
    dev_yhat_raw_1 = dev_yhat_raw.reshape(-1)
    dev_y_1 = dev_y.reshape(-1)
    sort_arg = np.argsort(dev_yhat_raw_1)
    sort_label = np.take_along_axis(dev_y_1, sort_arg, axis=0)
    label_count = np.sum(sort_label)
    correct = label_count - np.cumsum(sort_label)
    predict = dev_y_1.shape[0] + 1 - np.cumsum(np.ones_like(sort_label))
    f1 = 2 * correct / (predict + label_count)
    sort_yhat_raw = np.take_along_axis(dev_yhat_raw_1, sort_arg, axis=0)
    f1_argmax = np.argmax(f1)
    threshold = sort_yhat_raw[f1_argmax]
    return threshold


class SelfAttV4(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

        # 这样很清晰
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        # 一般是 0.1 的 dropout，一般写作 config.attention_probs_dropout_prob
        # hidden_dropout_prob 一般也是 0.1
        self.att_drop = nn.Dropout(0.1)

        # 可以不写；具体和面试官沟通。
        # 这是 MultiHeadAttention 中的产物，这个留给 MultiHeadAttention 也没有问题；
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, X, attention_mask=None):
        # attention_mask shape is: (batch, seq)
        # X shape is: (batch, seq, dim)

        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        att_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.dim)
        if attention_mask is not None:
            # 给 weight 填充一个极小的值
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))

        att_weight = torch.softmax(att_weight, dim=-1)
        # print(att_weight)

        att_weight = self.att_drop(att_weight)

        output = att_weight @ V
        ret = self.output_proj(output)
        return ret

class ClassifiyZYBERT(nn.Module):
    def __init__(self, Bertmodel_path, Bertmodel_config):
        super(ClassifiyZYBERT, self).__init__()
        self.PreBert = BertModel.from_pretrained(Bertmodel_path, config=Bertmodel_config)
        # for parameter in self.PreBert.parameters():
        #     parameter.requires_grad = False
        # self.PreBert1 = BertModel.from_pretrained(Bertmodel_path, config=Bertmodel_config)
        # for parameter in self.PreBert1.parameters():
        #     parameter.requires_grad = False

        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

        # self.syndrome_SelfAttention = SelfAttV4(768)
        self.clssificion_syndrome = nn.Linear(1024, 10, bias=True)

        self.first_linears = nn.Linear(1024, 512, bias=False)
        self.second_linears = nn.Linear(512, 10, bias=False)
        self.third_linears = nn.Linear(1024, 10, bias=True)

        self.clssificion_diease = nn.Linear(1024,4,bias=True)

    def forward(self, batch=None, token_type_ids=None, return_dict=None):
        input_ids = batch[0]
        attention_mask = batch[1]
        y = batch[2]  #真实标签

        # input_ids = input_ids.reshape(-1, 512)
        # attention_mask = attention_mask.reshape(-1, 512)

        x_student = self.PreBert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 return_dict=return_dict)

        # h_classificy_syndrome = self.syndrome_SelfAttention(self.dropout(x_student[0]))

        # syndrome_yeict = self.clssificion_syndrome(self.dropout(x_student[1]))
        #使用laat
        weight = F.tanh(self.first_linears(self.dropout(x_student[0])))
        attention_weight = self.second_linears(weight)
        attention_weight = F.softmax(attention_weight,1).transpose(1,2)
        weight_output = attention_weight @ x_student[0]
        syndrome_yeict = self.third_linears.weight.mul(weight_output).sum(dim=2).add(self.third_linears.bias)


        diease_yeict = self.clssificion_diease(self.dropout(x_student[1]))

        # threshold = find_threshold_micro(h_classificy_syndrome.cpu().detach().numpy(), y.cpu().detach().numpy())
        # yhat_syndrome = syndrome_yeict >= threshold

        # yhat_syndrome = self.sigmoid(syndrome_yeict) >= 0.9

        # 使用 torch.topk 获取每行的前两大值
        topk_values, topk_indices = torch.topk(syndrome_yeict, 2, dim=1)
        # 获取最大值和次大值
        top_max_values = topk_values[:, 0]  # 第一大值
        second_max_values = topk_values[:, 1]  # 第二大值
        # 对于每个 batch，检查最大值和次大值的差值
        diff = top_max_values - second_max_values
        # 如果差值大于0.2，则将次大值设置为最大值
        second_max_values[diff > 0.2] = top_max_values[diff > 0.2]
        yhat_syndrome = torch.eq(syndrome_yeict, second_max_values.unsqueeze(1))

        max_values, _ = torch.max(diease_yeict, dim=1, keepdim=True)
        # 将最大值的位置设置为True，其余位置为False
        yhat_diease = torch.eq(diease_yeict, max_values)

        return {"yhat_raw_syndrome": syndrome_yeict, "yhat_raw_diease": diease_yeict, "yhat_syndrome": yhat_syndrome, "yhat_diease": yhat_diease, "y": y}


model_path = './ZY_BERT'
model_config = BertConfig.from_pretrained('./ZY_BERT/config.json')
tokenizer = BertTokenizer.from_pretrained(model_path)
model = ClassifiyZYBERT(Bertmodel_path=model_path, Bertmodel_config=model_config)
model = model.to('cuda:0')
# gpus = [0, 1]
# model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict = TCM_SD_Data_Loader(tokenizer)

lr = 1e-6
optimizer = AdamW(model.parameters(),
                  lr=lr,  # args.learning_rate - default is 5e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8
                  )
total_steps = len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)
epochs = 15
criterions = nn.CrossEntropyLoss()
syndrome_criterions = nn.BCEWithLogitsLoss()

best_micro_metric = 0
best_epoch = 0

for epoch_i in range(1, epochs + 1):
    model.train()
    model.zero_grad()
    sum_loss = 0
    outputs = []
    t_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for step, batch in t_bar:
        batch = [tensor.to('cuda:0') for tensor in batch]

        now_res = model(batch=batch, token_type_ids=None, return_dict=False)
        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})

        # #总体标签进行计算loss
        # label = batch[2].float()
        # xx = now_res['yhat_raw']
        # loss = criterions(xx, label)

        #计算证型loss
        syndrome_label = batch[2][:,:10].float()
        yhat_raw_syndrome = now_res['yhat_raw_syndrome']
        syndrome_loss = criterions(yhat_raw_syndrome, syndrome_label)
        #计算疾病loss
        diease_label = batch[2][:,10:].float()
        yhat_raw_diease = now_res['yhat_raw_diease']
        diease_loss = criterions(yhat_raw_diease, diease_label)

        total_loss =  syndrome_loss + diease_loss
        sum_loss += total_loss
        avg_loss = sum_loss / step

        t_bar.update(1)  # 更新进度
        t_bar.set_description("avg_total_loss:{}, syndrome_loss:{}, diease_loss:{}".format(avg_loss, syndrome_loss, diease_loss))  # 更新描述
        t_bar.refresh()  # 立即显示进度条更新结果

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)


    #预测证型
    yhat_raw_syndrome = torch.cat([output['yhat_syndrome'].to(torch.int) for output in outputs]).cpu().detach().numpy()
    y_syndrome = torch.cat([output['y'][:,:10] for output in outputs]).cpu().detach().numpy()
    #预测疾病
    yhat_raw_diease = torch.cat([output['yhat_diease'].to(torch.int) for output in outputs]).cpu().detach().numpy()
    y_diease = torch.cat([output['y'][:,10:] for output in outputs]).cpu().detach().numpy()
    # 使用 np.all 逐行比较证型的正确率
    comparison_syndrome = np.all(yhat_raw_syndrome == y_syndrome, axis=1)
    matching_rows_count_syndrome = np.sum(comparison_syndrome)
    ACC_syndrome = matching_rows_count_syndrome / yhat_raw_syndrome.shape[0]
    # 使用 np.all 逐行比较疾病的正确率
    comparison_diease = np.all(yhat_raw_diease == y_diease, axis=1)
    matching_rows_count_diease = np.sum(comparison_diease)
    ACC_diease = matching_rows_count_diease / y_diease.shape[0]
    #ACC是syndrome acc和diease acc的平均值
    total_ACC = (ACC_syndrome + ACC_diease) / 2
    print('Train:-----------Total ACC:{}, syndrome ACC:{}, diease ACC:{}'.format(total_ACC, ACC_syndrome, ACC_diease))



    model.eval()
    outputs = []
    for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Dev'):
        batch = [tensor.to('cuda:0') for tensor in batch]
        with torch.no_grad():
            now_res = model(batch=batch, token_type_ids=None, return_dict=False)
        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})
    #预测证型
    yhat_raw_syndrome = torch.cat([output['yhat_syndrome'].to(torch.int) for output in outputs]).cpu().detach().numpy()
    y_syndrome = torch.cat([output['y'][:,:10] for output in outputs]).cpu().detach().numpy()
    #预测疾病
    yhat_raw_diease = torch.cat([output['yhat_diease'].to(torch.int) for output in outputs]).cpu().detach().numpy()
    y_diease = torch.cat([output['y'][:,10:] for output in outputs]).cpu().detach().numpy()
    # 使用 np.all 逐行比较证型的正确率
    comparison_syndrome = np.all(yhat_raw_syndrome == y_syndrome, axis=1)
    matching_rows_count_syndrome = np.sum(comparison_syndrome)
    ACC_syndrome = matching_rows_count_syndrome / yhat_raw_syndrome.shape[0]
    # 使用 np.all 逐行比较疾病的正确率
    comparison_diease = np.all(yhat_raw_diease == y_diease, axis=1)
    matching_rows_count_diease = np.sum(comparison_diease)
    ACC_diease = matching_rows_count_diease / y_diease.shape[0]
    #ACC是syndrome acc和diease acc的平均值
    total_ACC = (ACC_syndrome + ACC_diease) / 2
    print('Dev:------------Total ACC:{}, syndrome ACC:{}, diease ACC:{}'.format(total_ACC, ACC_syndrome, ACC_diease))



    model.eval()
    outputs = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader),desc='Test'):
        batch = [tensor.to('cuda:0') for tensor in batch]
        with torch.no_grad():
            now_res = model(batch=batch, token_type_ids=None, return_dict=False)
        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})
    #预测证型
    yhat_raw_syndrome = torch.cat([output['yhat_syndrome'].to(torch.int) for output in outputs]).cpu().detach().numpy()
    # print(yhat_raw_syndrome)
    y_syndrome = torch.cat([output['y'][:,:10] for output in outputs]).cpu().detach().numpy()
    #预测疾病
    yhat_raw_diease = torch.cat([output['yhat_diease'].to(torch.int) for output in outputs]).cpu().detach().numpy()
    y_diease = torch.cat([output['y'][:,10:] for output in outputs]).cpu().detach().numpy()
    print(yhat_raw_diease)
    # 使用 np.all 逐行比较证型的正确率
    comparison_syndrome = np.all(yhat_raw_syndrome == y_syndrome, axis=1)
    matching_rows_count_syndrome = np.sum(comparison_syndrome)
    ACC_syndrome = matching_rows_count_syndrome / yhat_raw_syndrome.shape[0]
    # 使用 np.all 逐行比较疾病的正确率
    comparison_diease = np.all(yhat_raw_diease == y_diease, axis=1)
    matching_rows_count_diease = np.sum(comparison_diease)
    ACC_diease = matching_rows_count_diease / y_diease.shape[0]


    # ACC_syndrome = accuracy_score(y_syndrome, yhat_raw_syndrome)
    # ACC_diease = accuracy_score(y_diease, yhat_raw_diease)

    #ACC是syndrome acc和diease acc的平均值
    total_ACC = (ACC_syndrome + ACC_diease) / 2
    print('Test:------------Total ACC:{}, syndrome ACC:{}, diease ACC:{}'.format(total_ACC, ACC_syndrome, ACC_diease))


    # if micro_metric > best_micro_metric:
    #     torch.save(model,
    #                '../save/model_best_Chinese_WWM_{}.pkl'.format(args.syndrome_diag))
    #     best_micro_metric = micro_metric
    #     best_epoch = epoch_i
# print(best_epoch)
