from tqdm import tqdm
import json
import torch
import os
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def TCM_SD_Data_Loader(tokenizer):
    #找到所有的标签
    # syndromes = ['气虚血瘀证', '痰瘀互结证', '气阴两虚证', '气滞血瘀证', '肝阳上亢证', '阴虚阳亢证', '痰热蕴结证', '痰湿痹阻证', '阳虚水停证', '肝肾阴虚证','胸痹心痛病', '心衰病', '眩晕病', '心悸病']   #使用bert进行预测疾病和证型联合时标签
    syndromes = ['胸痹心痛病', '心衰病', '眩晕病', '心悸病']        #单使用bert进行预测疾病时标签
    # syndromes = ['气虚血瘀证', '痰瘀互结证', '气阴两虚证', '气滞血瘀证', '肝阳上亢证', '阴虚阳亢证', '痰热蕴结证', '痰湿痹阻证', '阳虚水停证', '肝肾阴虚证']  #单使用bert进行预测证型时标签


    id2syndrome_dict = {}
    syndrome2id_dict = {}

    y = 0
    for i in range(len(syndromes)):
        id2syndrome_dict[i] = syndromes[i]
        syndrome2id_dict[syndromes[i]] = i

    def get_InputTensor(path):

        contents = []
        with open(path, 'r', encoding='utf-8') as file:
            file = json.load(file)
            for line in file:
                contents.append(line)

        labels = []
        input_ids = []
        attention_masks = []
        true_splitNumbers = []
        sentences = []

        for content in tqdm(contents, desc='Loading data',total=len(contents)):

            sentence = content['症状'] + content['中医望闻切诊']
            sentences.append(sentence)
            labele_sentence = [0]*(len(syndromes))
            # 证型标签
            for label in content['证型'].split('|'):
                id = syndrome2id_dict[label]
                labele_sentence[id] = 1
            # 疾病标签
            labele_sentence[syndrome2id_dict[content['疾病']]] = 1
            labels.append(torch.tensor(labele_sentence))

        # 输入数据集种的文本
        input_ids_sen = []
        attention_masks_sen = []
        for sentence in tqdm(sentences,desc='Loading clinic text {} sentence'.format(path)):
            sentencei = sentence[:512]
            encoded_dicti = tokenizer(
                sentencei,  # 输入文本
                add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                max_length=100,  # 填充 & 截断长度
                padding='max_length',
                return_attention_mask=True,  # 返回 attn. masks.
                return_tensors='pt',  # 返回 pytorch tensors 格式的数据
                truncation=True
            )
            input_idsi = encoded_dicti['input_ids'][0].reshape(1,-1)
            attention_maski = encoded_dicti['attention_mask'][0].reshape(1,-1)
            input_ids_sen.append(input_idsi)
            attention_masks_sen.append(attention_maski)
        input_ids = torch.cat(input_ids_sen, dim=0)
        attention_masks = torch.cat(attention_masks_sen, dim=0)
        labels = torch.stack(labels, dim=0)
        return input_ids, attention_masks, labels

    input_ids_train, attention_masks_train, labels_train = get_InputTensor(
        './dataset/TCM-TBOSD-train.json')
    input_ids_test, attention_masks_test, labels_test = get_InputTensor(
        './dataset/TCM-TBOSD-test-A.json')
    input_ids_val, attention_masks_val, labels_val = get_InputTensor(
        './dataset/TCM-TBOSD-test-B.json')

        # 将输入数据合并为 TensorDataset 对象
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    val_dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # 为训练和验证集创建 Dataloader，对训练样本随机洗牌
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,  # 训练样本
        sampler=RandomSampler(train_dataset),  # 随机小批量
        batch_size=2,  # 以小批量进行训练
        drop_last=True,
    )

    # 测试集不需要随机化，这里顺序读取就好
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,  # 验证样本
        sampler=SequentialSampler(test_dataset),  # 顺序选取小批量
        batch_size=2,
        drop_last=True,
    )

    # 验证集不需要随机化，这里顺序读取就好
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,  # 验证样本
        sampler=SequentialSampler(val_dataset),  # 顺序选取小批量
        batch_size=2,
        drop_last=True
    )


    return train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict


# from transformers import BertTokenizer
# from transformers import BertConfig, BertModel, AdamW
# tokenizer = BertTokenizer.from_pretrained('./chinese_wwm_pytorch')
# TCM_SD_Data_Loader(tokenizer)