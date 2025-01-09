# 中医辨证辨病及中药处方生成测评

任务组织单位：齐鲁工业大学（山东省科学院）、山东中医药大学附属医院

任务组织人员：王聪、赵直倬、管红娇、鹿文鹏、王怡斐

中医作为中国传统医学的重要组成部分，历经数千年的发展，已形成独具特色的理论体系和诊疗方法，对中国乃至全球人民的医疗健康做出了重要贡献。辨证论治是中医认识疾病和治疗疾病的核心原则和方法，其基本思想是通过望、闻、问、切的方法，收集患者症状、舌苔、脉象等临床信息，通过分析、综合，辨清疾病的病因、病机，概括、判断为某种性质的证，进而制定个性化的治疗方案，开具合适的中药处方予以治疗。

为了推动人工智能在中医领域的应用、推动中医现代化的发展，本任务构建了一个新的用于评估中医辨证辨病及处方生成的数据集。该数据集基于脱敏病历数据而构建，共涉及10种中医证型（下称证型）、4种中医疾病（下称疾病）、381种中药，共计1500条数据。任务旨在评估辨证论治的算法性能，包括两个子任务：

子任务1：中医多标签辨证辨病

基于给定的患者临床文档，判断患者所患的证型和疾病。具体需要参考任务网址。

子任务2：中药处方推荐

基于给定的患者临床文档，为患者推荐合适的中药处方。具体需要参考任务网址。

# 一、数据集介绍

评测数据基于医院脱敏病历构建，共1500条数据。数据分为训练集、验证集和测试集，数据量分别为800、200和500。本任务仅公开训练集数据和无标签的验证集数据，测试集数据不公开。

## 数据集申请

1.   下载《数据使用与保密承诺书》在文档末尾填写参赛队伍信息，下载地址：。
2.   参赛队伍负责人签名（手写签名）。
3.   将签名的《数据使用与保密承诺书》扫描件（PDF）发送至以下邮箱tcmtbosd@163.com，邮件标题为：“参赛单位-队伍名称-中医辨证辨病及中药处方生成测评数据使用申请”。

## 标注数据的字段信息说明

-   ID：患者入院的唯一id
-   性别：男或女
-   职业：患者的职业信息，如职员、退（离）休人员等
-   年龄：患者的年龄。
-   婚姻：描述婚姻状况，如已婚、未婚等
-   病史陈述者：入院时描述患者身体状况的人员与患者本人的关系，如患者本人
-   发病节气：患者出现病情时所处于的节气，如清明、小雪等
-   主诉：患者在就诊时向医生描述的最主要、最直接的不适或症状，用一句简短的文本概括描述，通常是患者就医的主要原因
-   症状：患者入院时所表现出的主要症状和体征的概述
-   中医望闻切诊：医师对患者进行“望”、“闻”、“切”后，对患者状态的描述
-   病史：包括现病史、既往史、个人史、婚育史、家族史
-   体格检查：患者的体格检查
-   辅助检查：患者的其他检查项目，如CT、心电图报告等
-   疾病：患者对应中医的疾病，如心悸病、胸痹心痛病等
-   证型：患者对应的中医证型，如气虚血瘀证、痰热蕴结证等
-   处方（不包括剂量）：患者中药处方，如黄芪、白芷等

## 数据样例

```json
{
    "ID": "35",
    "性别": "女",
    "职业": "退休",
    "年龄": "66岁",
    "婚姻": "已婚",
    "病史陈述者": "本人",
    "发病节气": "立夏",
    "主诉": "发作性胸闷20年，加重伴胸痛3月余",
    "症状": "胸部疼痛，呈针刺样，胸闷不舒，心慌不安，气短乏力，眼干眼涩，口干口苦，纳可，食后反酸烧心，眠可，二便调。",
    "中医望闻切诊": "中医望闻切诊：表情自然，面色暗红，形体正常,动静姿态，语气低,气息平；无异常气味,舌暗红、苔黄腻,舌下脉络曲张,脉弦。",
    "病史": "现病史：患者20年前因劳累后出现胸闷，无胸痛，于“***”就诊，行心电图、心脏彩超等检查，诊断为“冠心病”，具体治疗不详，好转后出院。院外规律服用“拜阿司匹林”等，症状控制可。3月前劳累后出现上述症状加重，自服“复方丹参滴丸”，效不佳。后患者于“***就诊”，行冠脉CT示：“LM轻度狭窄，LAD中度狭窄，LCX轻度狭窄，RCA轻度狭窄，PDA中度狭窄”，予“吲哚布芬”、“喜格迈”等，效不佳。现为求进一步中西医结合系统治疗，入住我病区。入院症见：既往史：否认慢性支气管炎等慢性疾病病史。否认肝炎、否认结核等传染病史。预防接种史不详。否认手术史、否认重大外伤史。否认输血史。否认药物过敏史、否认其他接触物过敏史。个人史：生于******，久居本地，无疫水、疫源接触史，无嗜酒史，无吸烟史，无放射线物质接触史，否认麻醉毒品等嗜好，否认冶游史，否认食物过敏史，否认传染病史。婚育史：已婚，适龄婚育。月经史：已绝经，既往月经规律。家族史：父母已故，死因不详。兄弟姐妹6人，均体健。育有1子，儿子及配偶均体健，家人体健，否认家族性遗传病史。",
    "体格检查": "体温：36.5℃  脉搏：61次/分  呼吸：18次/分  血压：158/71mmHg（R）、152/71mmHg（L） Padua评分：3分（低危）生命体征一般情况：患者老年，女,发育正常，营养良好，神志清楚，步入病房,查体合作，皮肤黏膜：全身皮肤及粘膜无黄染,未见皮下出血,淋巴结浅表淋巴结未及肿大。标题定位符头颅五官无畸形，眼睑无水肿,巩膜无黄染,双侧瞳孔等大等圆，对光反射灵敏，外耳道无异常分泌物，鼻外观无畸形，口唇红润，伸舌居中，双侧扁桃体正常，表面未见脓性分泌物，标题定位符颈软，无抵抗感，双侧颈静脉正常，气管居中，甲状腺未及肿大，未闻及血管杂音。标题定位符胸廓正常,双肺呼吸音清晰，未闻及干、湿罗音，未闻及胸膜摩擦音。心脏心界不大，心率61次/分，心律齐整，心音低,各瓣膜听诊区未闻及杂音，未闻及心包摩擦音。脉搏规整，无水冲脉、枪击音、毛细血管搏动征。腹部腹部平坦，无腹壁静脉显露，无胃肠型和蠕动波，腹部柔软，无压痛、反跳痛，肝脏未触及，脾脏未触及，未触及腹部包块，麦氏点无压痛及反跳痛，Murphy's征－，肾脏未触及，肝浊音界正常，肝肾区无明显肾区叩击痛，肝脾区无明显叩击痛，腹部叩诊鼓音，移动性浊音-，肠鸣音正常，无过水声，直肠肛门、生殖器肛门及外生殖器未查。生理反射存在，病理反射未引出，双下肢无水肿。",
    "辅助检查": " 2020-4-29 冠脉CT示：LM轻度狭窄，LAD中度狭窄，LCX轻度狭窄，RCA轻度狭窄，PDA中度狭窄。（于***）2020-5-12 心电图示：窦性心律，ST-T改变。",
    "疾病": "胸痹心痛病",
    "证型": "气虚血瘀证|痰热蕴结证",
    "处方": " ['丁香', '广藿香', '黄芪', '檀香', '砂仁', '木香', '草豆蔻', '附片 ', '花椒', '制川乌', '细辛', '桔梗', '麸炒枳壳', '葛根']"
}
```

# 二、子任务1：中医多标签辨证辨病（TCM Mulit-label Syndrome and Disease Differentiation）

## 背景介绍

辨病是对患者的病因病机、病情发展等整体的把握，而辨证更加注重根据病情某一发展阶段的病理特点做出阶段性判断。

辨证是中医学的核心任务之一，是实施个性化诊疗的关键环节。中医要求医师不仅关注疾病的表面症状，还需要深入分析患者的整体状态和内在联系。这也体现在辨证过程中，即主证是疾病的核心表现，是判断病情性质和确定治疗方向的主要依据；而兼证则是疾病发展过程中伴随的其他次要表现。例如患者的主要表现为气短乏力、面色苍白等症状及相应的舌象（舌紫暗）、脉象（涩滞）等，指向气虚血瘀证（主证），而患者又同时出现肢体沉重、关节疼痛、舌苔白腻等，提示患者存在痰湿痹阻证（兼证）。这种主证与兼证的结合不仅反映了病情的复杂性，还进一步揭示了疾病的全貌。

辨别疾病、主证、兼证是一个复杂的推理过程，对医生的知识储备和推理能力有极高的要求，三者相互影响，增加了医生诊疗的难度。这对智能辅助诊疗技术提出了迫切要求。

## 任务描述

子任务1的目标：借助自然语言处理技术在从病历和症状描述中提取信息，快速分析主证、兼证和疾病的关系，提高辨病辨证的效率和准确性，辅助医生更精准地完成诊断。

子任务1的定义：给定一段由自然语言文本书写的患者的详细情况描述（包括现病史、主诉、四诊信息等），模型需要预测出患者所患的证型和疾病。

## 任务说明

该任务给定一段由临床信息构成的文本作为输入，需要模型输出对应的中医证型、疾病。

*   输入：诊疗记录中各类临床信息构成的文本，最终拼接成一个str类型字段
*   输出：对应的证型和疾病

本数据集候选的全部证型和疾病类别如下：

```
证型：['气虚血瘀证', '痰瘀互结证', '气阴两虚证', '气滞血瘀证', '肝阳上亢证', '阴虚阳亢证', '痰热蕴结证', '痰湿痹阻证', '阳虚水停证', '肝肾阴虚证']
疾病：['胸痹心痛病', '心衰病', '眩晕病', '心悸病']
```

注：输出为列表格式，其中所有的证型由‘|’符号隔开，并将疾病作为列表的第二项，如：[证型名1|证型名2,疾病名]。证型有一或多个，疾病有且仅有一个。

## 评测指标

中医多标签辨证辨病任务采用以下评测指标：

1.   辨证任务准确率（acc of syndrome differentiation）

     计算公式如下：
     
     
     $$syndrome_{acc} = \frac{NUM(y \cap \hat{y} )}{NUM(y)}$$
     
     
     其中， $y$ 是数据集样本中真实证型的列表和 $\hat{y}$ 是模型预测的数据集样本中的证型列表； $NUM(x)$ 代表数量函数，用来计算 $x$ 的数量。

2.   辨病任务准确率（acc of disease differentiation）

     计算公式如下：

     $$disease_{acc} = \frac{NUM(y \cap \hat{y} )}{NUM(y)}$$

     其中， $y$ 是数据集样本中真实疾病的列表和 $\hat{y}$ 是模型预测的数据集样本中的疾病列表； $NUM(x)$ 代表数量函数，用来计算$x$的数量。

3.   评价总指标
     
     $$task1 \textunderscore acc = \frac{1}{2}(syndrome_{acc} + disease_{acc})$$
     

# 三、子任务2：中药处方推荐（TCM Prescription Recommendation）
## 背景介绍

在实际诊疗过程中，医生首先根据患者四诊信息判断出“疾病”、“证型”，然后在辨证审因，确定治法的基础上，按照组方原则，选择恰当的药物合理配伍，组成中药处方。自然语言处理技术能够从大量的临床数据中提取有价值的信息，帮助医生快速、准确地进行辨证论治，并推荐个性化的中药处方。

## 任务描述

子任务2的目标：根据患者的详细情况描述，自动推荐一组中草药作为处方（不包括中药剂量）。

子任务2的定义：给定一段由自然语言文本书写的患者的详细情况描述（包括现病史、主诉、四诊信息等），模型推荐当前患者的中药处方（不包括中药剂量）。

## 任务说明

给定患者的基本信息和健康情况，给患者推荐一组中草药用作中药处方。

*   输入：诊疗记录中各类临床信息构成的文本
*   输出：中药处方，一组中草药的集合

本数据集涉及的全部候选中草药的集合如下：

```json
['冬瓜皮', '沉香', '茜草炭', '浮小麦', '炙甘草', '炒白扁豆', '砂仁', '合欢花', '北刘寄奴', '炒六神曲', '炒决明子', '益母草', '酒苁蓉', '炒僵蚕', '稀莶草', '秦艽', '黄酒', '瞿麦', ' 白鲜皮', '熟地黄', '扁蓄', '诃子肉', '煅牡蛎', '鸡血藤', '党参', '瓜蒌', '莲子', '酒五味子', '金钱草', '法半夏', '北败酱草', '花椒', '吴茱萸(粉)', '桑白皮', '茯神', '桂枝', '降香', '制远志', '琥珀', '佛手', '麦芽', '水红花子', '金银花', '马鞭草', '半枝莲', '炮姜', '生酸枣仁', '盐补骨脂', '炒瓜蒌子', '珍珠母', '乌药', '茵陈', '地肤子', '酸枣仁', '槟榔', '大青叶', '人参片', '麸煨肉豆蔻', '蛤蚧', '路路通', '蝉蜕', '马勃', '香橼', '络石藤', '狗脊', '蜈蚣', '制川乌', '白扁豆花', '麻黄', '射干', '厚朴', '蜂蜜', '柏子仁', '炒谷芽', '蜜百合', ' 石菖蒲', '白薇', '续断', '炒川楝子', '黄连片', '绵萆薢', '鹿角胶', '翻白草', '羚羊角粉', '天麻', '山慈菇', '菊花', '炒芥子', '墨旱莲', '蜜枇杷叶', '川芎', '酒大黄', '焦山楂', '红曲', '山药', '牡蛎', '海藻', '夏枯草', '白前', '白芍', '茯苓皮', '煅自然铜', '附片 ', '土茯苓', '制何首乌', '炒莱菔子', '黄芩', '蒲黄', '紫石英', '透骨草', '绞股蓝', '泽泻', '甘松', ' 炒酸枣仁', '儿茶', '马齿苋', '太子参', '薏苡仁', '萹蓄', '青蒿', '苏木', '桑叶', '连翘', '穿山龙', '忍冬藤', '苦参', '炒茺蔚子', '防己', '益母草炭', '莲须', '猫眼草', '麸炒芡实', ' 炒牛蒡子', '龟甲胶', '蜜槐角', '柿蒂', '龙骨', '泽兰', '桔梗', '青葙子', '冰片', '大枣', '侧柏叶', '三七粉', '醋乳香', '川牛膝', '全蝎', '合欢皮', '首乌藤', '醋鳖甲', '炒蔓荆子', ' 烫骨碎补', '紫苏叶', '盐沙苑子', '南沙参', '石见穿', '胆南星', '焦白术', '酒黄芩', '白术', '鬼箭羽', '玫瑰花', '干姜', '牡丹皮', '白花蛇舌草', '酒当归', '火麻仁', '炒桃仁', '醋鸡内金', '磁石', '醋龟甲', '白茅根', '肉桂', '白及', '油松节', '炒苍耳子', '化橘红', '佩兰', '芦根', '紫草', '酒萸肉', '丹参', '柴胡', '制巴戟天', '木蝴蝶', '炒紫苏子', '浮萍', '栀子', '甘草片', '木香', '丝瓜络', '炒麦芽', '板蓝根', '车前草', '炒王不留行', '朱砂', '醋三棱', '辛夷', '土鳖虫', '煅龙骨', '炒白芍', '炒白果仁', '芒硝', '赭石', '西洋参', '桑枝', '红景天', '锁阳', '淫羊藿', '酒乌梢蛇', '制草乌', '肉苁蓉片', '麸炒枳壳', '炒苦杏仁', '炙黄芪', '黄连', '重楼', '细辛', '蜜旋覆花', '醋没药', '玉竹', '蛤壳', '草豆蔻', '炙淫羊藿', '广藿香', '麸炒枳实', '鱼腥草', '鹿角霜', '通草', '烫水蛭', '水牛角', '烫狗脊', '盐续断', '盐益智仁', '常山', '百部', '阿胶', '藁本片', '制吴茱萸', '豆蔻', '酒女贞子', '片姜黄', '蜜款冬花', '龙胆', '寒水石', '莲子心', '荷叶', '防风', '炒蒺藜', '川贝母', '虎杖', '海桐皮', '甘草', '赤石脂', '麻黄根', '郁金', '海风藤', '青皮', '地龙', '地榆', '石韦', '焦栀子', '盐杜仲', '清半夏', '盐知母', '薤白', '茜草', '荆芥炭', '百合', '龙齿', '石决明', '炒葶苈子', '知母', '赤小豆', '麸炒白术', '酒仙茅', '淡竹叶', '大黄', '海螵蛸', '仙鹤草', '白芷', '麸炒薏苡仁', '青风藤', '前胡', '升麻', '海浮石', '制天南星', '麸炒山药', '蒲公英', '豨莶草', '当归', '醋莪术', '薄荷', '红参片', '生地黄', '苦地丁', '炒槐米', '蜜桑白皮', '盐小茴香', '麸炒苍 术', '姜半夏', '钟乳石', '桑椹', '瓜蒌皮', '葛根', '桑螵蛸', '浙贝片', '菟丝子', '醋延胡索', '艾叶', '五加皮', '炒冬瓜子', '瓦楞子', '盐黄柏', '醋五灵脂', '石膏', '醋山甲', '檀香', '皂角刺', '红花', '野菊花', '木瓜', '蜜麻黄', '槲寄生', '密蒙花', '蜜百部', '蜜紫菀', '茯苓', '海金沙', '麦冬', '猪苓', '天竺黄', '石斛', '枸杞子', '徐长卿', '醋香附', '麸神曲', '黄芪', '郁李仁', '枯矾', '盐车前子', '伸筋草', '草果仁', '山楂', '炒稻芽', '威灵仙', '淡豆豉', '蛇莓', '丁香', '盐荔枝核', '绵马贯众', '黄柏', '独活', '覆盆子', '龙眼肉', '老鹳草', ' 乌梅', '紫苏梗', '制白附子', '大腹皮', '竹茹', '天花粉', '乌梅炭', '滑石粉', '冬葵子', '灯心草', '六月雪', '牛膝', '陈皮', '荆芥', '炒甘草', '北沙参', '地骷髅', '地骨皮', '赤芍', ' 玄参', '桑葚', '酒黄精', '羌活', '钩藤', '天冬']
```

注：输出为列表格式，如：[中草药名1, 中草药名2, 中草药名3]。

## 评测指标

1.   Jaccard相似系数（Jaccard Similarity Coefficient）

     Jaccard相似系数用于衡量两个集合的相似度，Jaccard相似系数的取值范围为 [0, 1]，值越大表示预测结果与真实标签的相似度越高。计算公式如下：

     $$Jaccard(y, \hat{y}) = \frac{NUM(y \cap \hat{y})}{NUM(y \cup \hat{y})}$$
     
     其中， $y$ 是真实处方， $\hat{y}$ 是模型预测的处方， $NUM(x)$ 代表数量函数，用来计算 $x$ 的数量。

2.   Recall

     Recall 用于衡量预测结果中，与真实标签匹配的数量占所有真实标签总数的比例。计算公式如下：
     
     $$Recall(y, \hat{y}) = \frac{NUM(y \cap \hat{y})}{NUM(y)}$$
     
     其中， $y$ 是真实处方， $\hat{y}$ 是模型预测的处方， $NUM(x)$ 代表数量函数，用来计算 $x$ 的数量。

3.   Precision

     Precision 用于衡量预测结果中，与真实标签匹配的数量占预测标签总数的比例。计算公式如下：

     $$Precision(y,\hat{y}) = \frac{NUM(y \cap \hat{y})}{NUM(\hat{y})}$$

     其中， $y$ 是真实处方， $\hat{y}$ 是模型预测的处方， $NUM(x)$ 代表数量函数，用来计算 $x$ 的数量。

4.   F1分数

     F1分数是Precision和Recall的调和平均数，用于综合衡量模型的准确性和召回率。计算公式如下：

     $${F1}(y, \hat{y}) = 2 \cdot \frac{\text{Precision}(y, \hat{y}) \cdot \text{Recall}(y, \hat{y})}{\text{Precision}(y, \hat{y}) + \text{Recall}(y, \hat{y})}$$

     其中， $y$ 是真实处方， $\hat{y}$ 是模型预测的处方。

5.   药物平均数量(Avg Herb)

     药物平均数量用于衡量模型推荐的中药方剂数量与真实标签数量的接近程度。计算方法是通过比较模型推荐的中药数量和真实标签的中药数量，并计算它们的匹配度。匹配度越高，表示模型推荐的中药数量越接近真实标签的数量。计算公式如下：
     
     $$AVG(y, \hat{y}) = 1 - \frac{\lvert NUM(y) - NUM(\hat{y}) \rvert}{max( NUM(y) , NUM(\hat{y}) )}$$
     
     其中， $y$ 是真实处方， $\hat{y}$ 是模型预测的处方。 $NUM(x)$ 代表数量函数，用来计算 $x$ 的数量， $max⁡(a,b)$ 代表取 $a$ , $b$ 中的最大值， $|x|$ 代表计算 $x$ 的绝对值。

6.   评价总指标
     
     $$task2 \textunderscore score = \frac{1}{3} \cdot \frac{1}{N}\sum_{i=1}^N{[Jaccard(y_i, \hat{y_i}) + {F1}(y_i, \hat{y_i}) + AVG(y_i, \hat{y_i})]}$$
     
     其中 $y_i$ 是第 $i$ 条样本的真实处方， $\hat{y_i}$ 是模型预测的第 $i$ 条样本的处方， $N$ 表示样本总数。

# 四、结果提交

本次测评分A榜（验证集）、B榜（测试集）两个榜单。

A榜评测结果采用邮件方式提交。邮件标题为：“CCL2025-中医辨证辨病及中药处方生成测评-参赛单位-队伍名称-A榜结果-子任务名称”。测试结果命名为： TCM-TBOSD-A.json。榜单每日17:59:59在Github中进行更新。具体格式如下方样例，当参加单个子任务时，另一子任务值设置为空列表：

```
[
    {
    "ID":1,
    "子任务1":[证型名1|证型名2,疾病名],
    "子任务2":[药物名1, 药物名2, 药物名3]
    },
]
```

B榜评测结果使用docker镜像提交，参赛队伍需要将模型打包成docker镜像，同时提交使用说明。镜像通过百度云盘上传，提交镜像压缩包名称命名：“CCL2025-TCM-TBOSD-参赛单位-队伍名称-子任务名称.tar”，同时使用邮件提交百度云盘下载链接，邮件标题为：“CCL2025-中医辨证辨病及中药处方生成测评-参赛单位-队伍名称-B榜结果-子任务名称”。

注：模型参数不能超过7B。 B榜为最终榜单。

# 五、系统排名

1.   所有测评任务均采用百分制分数显示，小数点后保留2位。

2.   系统排名取各项任务得分的加权和（两个子任务权重依次为0.5，0.5），即：
     
     $$task \textunderscore score = 0.5\cdot task1 \textunderscore acc + 0.5 \cdot task2 \textunderscore score$$
     
     如果某项任务未提交，默认分数为0，仍参与到系统最终得分的计算。

# 六、Baseline

Baseline下载地址：https://github.com/QLU-NLP/TCM-Syndrome-and-Disease-Differentiation-and-Prescription-Recommendation

Baseline表现：

| task1_acc | task2_score | task_score |
| :-------: | :---------: | :--------: |
|           |             |            |

# 七、评测数据

数据由json格式给出，数据集包含以下内容：

-   TCM-TBOSD-train.json: 训练集标注数据。
-   TCM-TBOSD-test-A.json: A榜测试集（验证集）。
-   TCM-TBOSD-A.json: A榜提交示例
-   TCM-TBOSD-test-B.json: B榜测试集（测试集）。B榜测试集不公开。

# 八、赛程安排

本次大赛分为报名组队、A榜、B榜三个阶段，具体安排和要求如下：

1.   报名时间：2025年2月10日-5月2日
2.   训练、验证数据及baseline发布：2025年2月10日
3.   测试A榜（验证集）数据发布：2025年2月10日
4.   测试A榜评测截止：2025年5月4日 17:59:59
5.   测试B榜（测试集）最终结果：2025年5月10日 17:59:59
6.   公布测试结果：2025年5月15日前
7.   提交中文或英文技术报告：2025年6月1日
8.   中文或英文技术报告反馈：2025年6月20日
9.   正式提交中英文评测论文：2025年7月1日
10.   公布获奖名单：2025年7月5日
11.   评测研讨会：2025年8月

**注意：报名组队与认证（2025年2月10日—5月2日）**

# 九、报名方式

2025年2月10日将开放本次比赛的报名组队，给tcmtbosd@163.com邮箱，发送个人信息（包括学校、团队名、队长、组员、队长的学生证明或队长的其他身份证明）以及《数据使用与保密承诺书》进行注册，即可报名参赛；选手可以单人参赛，也可以组队参赛。组队参赛的每个团队不超过5人，每位选手只能加入一支队伍；选手需确保报名信息准确有效，组委会有权取消不符合条件队伍的参赛资格及奖励；选手报名、组队变更等操作截止时间为5月2日23：59：59； 

向赛题举办方发送电子邮件进行报名，以获取数据解压密码。邮件标题为：“CCL2025-中医辨证辨病及中药处方生成测评-参赛单位”，例如：“CCL2025-中医辨证辨病及中药处方生成测评-齐鲁工业大学”；附件内容为队伍的参赛报名表，报名表点此下载，同时报名表应更名为“参赛队名+参赛队长信息+参赛单位名称”。请参加评测的队伍发送邮件至 tcmtbosd@163.com，报名成功后赛题数据解压密码会通过邮件发送给参赛选手。

# 十、赛事规则

1.   由于版权保护问题，TCM-TBOSD数据集只免费提供给用户用于非盈利性科学研究使用，参赛人员不得将数据用于任何商业用途。如果用于商业产品，请联系鹿文鹏、王怡斐老师，联系邮箱wenpeng.lu@qlu.edu.cn。

2.   每名参赛选手只能参加一支队伍，一旦发现某选手参加多支队伍，将取消相关队伍的参赛资格。

3.   数据集的具体内容、范围、规模及格式以最终发布的真实数据集为准，验证集不可用于模型训练。

4.   参赛队伍可在参赛期间随时上传测试集的预测结果，A榜阶段每天可提交1次、B榜阶段最多提交2次，本任务组织单位在每日17:59:59更新当前最新榜单排名情况，严禁参赛团队注册其它账号多次提交。

5.   允许使用公开的代码、工具、外部数据（从其他渠道获得的标注数据）等，但需要保证参赛结果可以复现。

6.   参赛队伍可以自行设计和调整模型，但需注意模型参数量最多不超过7B。

7.   算法与系统的知识产权归参赛队伍所有。要求最终结果排名前8名的队伍提供算法代码与系统报告（包括方法说明、数据处理、参考文献和使用的开源工具、外部数据等信息）。提交完毕将采用随机交叉检查的方法对各个队伍提交的模型进行检验，如果在排行榜上的结果无法复现，将取消获奖资格。

8.   参赛团队需保证提交作品的合规性，若出现下列或其他重大违规的情况，将取消参赛团队的参赛资格和成绩，获奖团队名单依次递补。重大违规情况如下：

     a. 使用小号、串通、剽窃他人代码等涉嫌违规、作弊行为；

     b. 团队提交的材料内容不完整，或提交任何虚假信息；

     c. 参赛团队无法就作品疑义进行足够信服的解释说明；

9.   获奖队伍必须注册会议并在线下参加（如遇特殊情况，可申请线上参加）。