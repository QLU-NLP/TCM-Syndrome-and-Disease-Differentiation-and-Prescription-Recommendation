import json
import random
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model_name = "/model/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

patients = []
patient_dict = {}
data_path = '/tcdata/TCM-TBOSD-test-B.json'
# output_path = f'/app/tmp/task2.jsonl'
# data_path = '/data/TCM-TBOSD-test-B.json'
output_path = f'/app/tmp/task2.jsonl'

data = json.load(open(data_path, 'r', encoding='utf-8'))
print(f'len(data): {len(data)}')

for idx, item in tqdm(enumerate(data), total=len(data)):
    # ['ID', '性别', '职业', '年龄', '婚姻', '病史陈述者', '发病节气', '主诉', '症状', '中医望闻切诊', '病史', '体格检查', '辅助检查', '疾病', '证型', '处方']
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
    query = f'''
任务：根据患者[主诉],[症状],[中医望闻切诊]等信息,在[草药]中为患者推荐需要使用的[推荐草药]。
# 要求1: 草药局限在下方列表中。
[草药]: '冬瓜皮', '沉香', '茜草炭', '浮小麦', '炙甘草', '炒白扁豆', '砂仁', '合欢花', '北刘寄奴', '炒六神曲', '炒决明子', '益母草', '酒苁蓉', '炒僵蚕', '稀莶草', '秦艽', '黄酒', '瞿麦', ' 白鲜皮', '熟地黄', '扁蓄', '诃子肉', '煅牡蛎', '鸡血藤', '党参', '瓜蒌', '莲子', '酒五味子', '金钱草', '法半夏', '北败酱草', '花椒', '吴茱萸(粉)', '桑白皮', '茯神', '桂枝', '降香', '制远志', '琥珀', '佛手', '麦芽', '水红花子', '金银花', '马鞭草', '半枝莲', '炮姜', '生酸枣仁', '盐补骨脂', '炒瓜蒌子', '珍珠母', '乌药', '茵陈', '地肤子', '酸枣仁', '槟榔', '大青叶', '人参片', '麸煨肉豆蔻', '蛤蚧', '路路通', '蝉蜕', '马勃', '香橼', '络石藤', '狗脊', '蜈蚣', '制川乌', '白扁豆花', '麻黄', '射干', '厚朴', '蜂蜜', '柏子仁', '炒谷芽', '蜜百合', ' 石菖蒲', '白薇', '续断', '炒川楝子', '黄连片', '绵萆薢', '鹿角胶', '翻白草', '羚羊角粉', '天麻', '山慈菇', '菊花', '炒芥子', '墨旱莲', '蜜枇杷叶', '川芎', '酒大黄', '焦山楂', '红曲', '山药', '牡蛎', '海藻', '夏枯草', '白前', '白芍', '茯苓皮', '煅自然铜', '附片 ', '土茯苓', '制何首乌', '炒莱菔子', '黄芩', '蒲黄', '紫石英', '透骨草', '绞股蓝', '泽泻', '甘松', ' 炒酸枣仁', '儿茶', '马齿苋', '太子参', '薏苡仁', '萹蓄', '青蒿', '苏木', '桑叶', '连翘', '穿山龙', '忍冬藤', '苦参', '炒茺蔚子', '防己', '益母草炭', '莲须', '猫眼草', '麸炒芡实', ' 炒牛蒡子', '龟甲胶', '蜜槐角', '柿蒂', '龙骨', '泽兰', '桔梗', '青葙子', '冰片', '大枣', '侧柏叶', '三七粉', '醋乳香', '川牛膝', '全蝎', '合欢皮', '首乌藤', '醋鳖甲', '炒蔓荆子', ' 烫骨碎补', '紫苏叶', '盐沙苑子', '南沙参', '石见穿', '胆南星', '焦白术', '酒黄芩', '白术', '鬼箭羽', '玫瑰花', '干姜', '牡丹皮', '白花蛇舌草', '酒当归', '火麻仁', '炒桃仁', '醋鸡内金', '磁石', '醋龟甲', '白茅根', '肉桂', '白及', '油松节', '炒苍耳子', '化橘红', '佩兰', '芦根', '紫草', '酒萸肉', '丹参', '柴胡', '制巴戟天', '木蝴蝶', '炒紫苏子', '浮萍', '栀子', '甘草片', '木香', '丝瓜络', '炒麦芽', '板蓝根', '车前草', '炒王不留行', '朱砂', '醋三棱', '辛夷', '土鳖虫', '煅龙骨', '炒白芍', '炒白果仁', '芒硝', '赭石', '西洋参', '桑枝', '红景天', '锁阳', '淫羊藿', '酒乌梢蛇', '制草乌', '肉苁蓉片', '麸炒枳壳', '炒苦杏仁', '炙黄芪', '黄连', '重楼', '细辛', '蜜旋覆花', '醋没药', '玉竹', '蛤壳', '草豆蔻', '炙淫羊藿', '广藿香', '麸炒枳实', '鱼腥草', '鹿角霜', '通草', '烫水蛭', '水牛角', '烫狗脊', '盐续断', '盐益智仁', '常山', '百部', '阿胶', '藁本片', '制吴茱萸', '豆蔻', '酒女贞子', '片姜黄', '蜜款冬花', '龙胆', '寒水石', '莲子心', '荷叶', '防风', '炒蒺藜', '川贝母', '虎杖', '海桐皮', '甘草', '赤石脂', '麻黄根', '郁金', '海风藤', '青皮', '地龙', '地榆', '石韦', '焦栀子', '盐杜仲', '清半夏', '盐知母', '薤白', '茜草', '荆芥炭', '百合', '龙齿', '石决明', '炒葶苈子', '知母', '赤小豆', '麸炒白术', '酒仙茅', '淡竹叶', '大黄', '海螵蛸', '仙鹤草', '白芷', '麸炒薏苡仁', '青风藤', '前胡', '升麻', '海浮石', '制天南星', '麸炒山药', '蒲公英', '豨莶草', '当归', '醋莪术', '薄荷', '红参片', '生地黄', '苦地丁', '炒槐米', '蜜桑白皮', '盐小茴香', '麸炒苍 术', '姜半夏', '钟乳石', '桑椹', '瓜蒌皮', '葛根', '桑螵蛸', '浙贝片', '菟丝子', '醋延胡索', '艾叶', '五加皮', '炒冬瓜子', '瓦楞子', '盐黄柏', '醋五灵脂', '石膏', '醋山甲', '檀香', '皂角刺', '红花', '野菊花', '木瓜', '蜜麻黄', '槲寄生', '密蒙花', '蜜百部', '蜜紫菀', '茯苓', '海金沙', '麦冬', '猪苓', '天竺黄', '石斛', '枸杞子', '徐长卿', '醋香附', '麸神曲', '黄芪', '郁李仁', '枯矾', '盐车前子', '伸筋草', '草果仁', '山楂', '炒稻芽', '威灵仙', '淡豆豉', '蛇莓', '丁香', '盐荔枝核', '绵马贯众', '黄柏', '独活', '覆盆子', '龙眼肉', '老鹳草', ' 乌梅', '紫苏梗', '制白附子', '大腹皮', '竹茹', '天花粉', '乌梅炭', '滑石粉', '冬葵子', '灯心草', '六月雪', '牛膝', '陈皮', '荆芥', '炒甘草', '北沙参', '地骷髅', '地骨皮', '赤芍', ' 玄参', '桑葚', '酒黄精', '羌活', '钩藤', '天冬'
# 要求2: 有多个中草药，每个中草药之间用逗号分隔。
# 要求3: 没有的中草药不要多写。
# 要求4: 输出中仅需要输出草药名称，不需要给出任何解释和其他信息，数量控制在10-15个左右。

例子：
[主诉]:患者主诉为xx,伴有xx。
[症状]:患者有xx,xx,xx等症状。
[中医望闻切诊]:舌苔xx,脉象xx。
[推荐草药]:
xxx, xxx, xxx, xxx, xxx, xxx, xxx, xxx

[主诉]:{chief_complaint}
[症状]:{symptom}
[中医望闻切诊]:{tcm_examination}
[推荐草药]:
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
    herb_list = response.split(',')
    data_item = {
        'ID':id,
        '子任务2':herb_list,
    }
    print(data_item)
    writer.write(json.dumps(data_item, ensure_ascii=False) + '\n')
    writer.close()