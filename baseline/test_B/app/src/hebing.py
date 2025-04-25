herb_file = '/app/tmp/task2.jsonl'
diease_file = '/app/tmp/task1_diease.jsonl'
syndrome_file = '/app/tmp/task1_syndrome.jsonl'

import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data 

herb_data = read_jsonl(herb_file)
diease_data = read_jsonl(diease_file)
syndrome_data = read_jsonl(syndrome_file)

output_file = '/app/result.json'
output_data = []

for i in range(len(herb_data)):
    ID = herb_data[i]['ID']
    herb = herb_data[i]['子任务2']
    temp = []
    for item in herb:
        temp.append(item.strip())
    herb = str(temp)
    diease = str(diease_data[i]['疾病'])
    syndrome = str(syndrome_data[i]['证型'])
    # Create a new dictionary for the output
    combined_data = {
        'ID': ID,
        '子任务1':[
            syndrome,
            diease
        ],
        '子任务2': herb
    }
    # Append the combined data to the output list
    output_data.append(combined_data)

json_data = json.dumps(output_data, ensure_ascii=False, indent=4)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json_data)
print(f"Combined data has been written to {output_file}")