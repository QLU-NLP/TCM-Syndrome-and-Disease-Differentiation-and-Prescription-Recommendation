import json
import random
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model_name = "/model/Qwen2.5-3B-Instruct"
# model_name = "/app/Qwen2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

patients = []
patient_dict = {}
data_path = '/tcdata/TCM-TBOSD-test-B.json'
# output_path = f'/app/tmp/task1_diease.jsonl'
# data_path = '/data/TCM-TBOSD-test-B.json'
output_path = f'/app/tmp/task1_diease.jsonl'
data = json.load(open(data_path, 'r', encoding='utf-8'))
print(f'len(data): {len(data)}')
for idx, item in tqdm(enumerate(data), total=len(data)):
    id = item['ID'] # id
    gender = item['性别'] # 性别
    job = item['职业'] # 职业
    age = item['年龄'] # 年龄
    marriage = item['婚姻'] # 婚姻
    status = item['病史陈述者']
    disease_time = item['发病节气']
    chief_complaint = item['主诉']
    symptom = item['症状'] # 症状
    tcm_examination = item['中医望闻切诊'] # 中医望闻切诊
    history = item['病史'] # 病史
    physical_examination = item['体格检查'] # 体格检查
    auxiliary_examination = item['辅助检查'] # 辅助检查
    drug = item['处方'] # 处方
    writer = open(output_path, 'a', encoding='utf-8')
    # 这里可以添加模型推理的代码
    query = f'''
任务：根据患者[主诉],[症状],[中医望闻切诊]等信息,在[疾病]中为患者预测疾病[疾病]。
# 要求1: 疾病局限在下方列表中。
[疾病]: 胸痹心痛病, 心衰病, 眩晕病, 心悸病
# 要求2: 只需要给出疾病名称,不需要给出疾病的描述。

例子：
[主诉]:患者主诉为xx,伴有xx。
[症状]:患者有xx,xx,xx等症状。
[中医望闻切诊]:舌苔xx,脉象xx。
[疾病]:
xx病

[主诉]:{chief_complaint}
[症状]:{symptom}
[中医望闻切诊]:{tcm_examination}
[疾病]:
'''

    messages = []
    messages.append({
        'role': 'user',
        'content': query
    })
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        top_k=50,
        top_p=0.8,
        # repetition_penalty=1.05,
        temperature=0.7
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    data_item = {
        'ID':id,
        '疾病':response,
    }
    print(data_item)
    writer.write(json.dumps(data_item, ensure_ascii=False) + '\n')
    writer.close()