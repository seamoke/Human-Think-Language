import json
total_data = []
with open('/mnt/bn/music-nas-dxj1/lil/new_MAmmoth/MAmmoTH/math_eval/dataset/gsm8k_tune.json', 'r') as f:
    tune_list = json.load(f)
    print(f'gsm8k :{len(tune_list)}')
with open('/mnt/bn/music-nas-dxj1/lil/new_MAmmoth/MAmmoTH/math_eval/dataset/math_tune.json', 'r') as f:
    temp_list = json.load(f)
    tune_list += temp_list
    print(f'math :{len(temp_list)}')
with open('/mnt/bn/music-nas-dxj1/lil/new_MAmmoth/MAmmoTH/math_eval/dataset/my_numglue.json', 'r') as f:
    temp_list = json.load(f)
    tune_list += temp_list
    print(f'numglue :{len(temp_list)}')
data_list = list(range(len(data)))
random.shuffle(data_list)
data = [sources[x] for x in data_list]
targets = [targets[x] for x in data_list]
# 对数据进行操作
# print(data[0])
# new_data = []
# for example in data:
#     pot,cot=example['COT'],example['POT']
#     if 'print(' in cot:
#         continue
#     new_data.append({'question':example['question'],'output':cot+"\n****\n"+pot})
# print(len(new_data))
# with open('format_tune.json', 'w') as f:
#     json.dump(new_data, f)
# print(data[0])
# new_data = []
# for example in data:
#     cot,pot=example['COT'],example['POT']
#     if 'print(' in cot:
#         continue
#     new_data.append({'question':example['question'],'output':cot+"\n****\n"+pot})
# print(len(new_data))
# with open('format_tune.json', 'w') as f:
#     json.dump(new_data, f)