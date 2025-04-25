import json
def read_jsonl(file_path):
    print('读取文件：', file_path)
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
name = 'LoRA5-testA'
file = f'/home/wangcong24/CCL2025/output/{name}.jsonl'
data = read_jsonl(file)
# jsonl to json
output = []
for item in data:
    herbs = item['子任务2']
    for i in range(len(herbs)):
        herbs[i] = herbs[i].replace(' ', '')
        herbs[i] = herbs[i].replace("'", "")
        herbs[i] = herbs[i].replace('"', '')
        herbs[i] = herbs[i].replace('[推荐草药]:', '')
        herbs[i] = herbs[i].strip()
    herbs = list(set(herbs))
    item['子任务2'] = str(herbs)
    output.append(item)
output_file = f'/home/wangcong24/CCL2025/output/{name}.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)