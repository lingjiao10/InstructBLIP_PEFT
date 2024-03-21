import json
import os

# with open('problems.json', 'r') as f:
with open('H:/Datasets/iconqa_data/problems.json', 'r') as f:
    data = json.load(f)

train, test, val = [], [], []

for key, value in data.items():
    split = value["split"]
    ques_type = value["ques_type"]
    if ques_type == "choose_txt":
        data = value
        data['id'] = key
        if split == "train" :
            train.append(data)
        elif split == "test":
            test.append(data)
        elif split == "val":
            val.append(data)

# anno_root = "/input/iconqa/annotations"
anno_root = 'H:/Datasets/iconqa_data/annotations'
if not os.path.exists(anno_root):
    os.mkdir(anno_root)

with open(anno_root + '/train.json', 'w') as train_file:
    json.dump(train, train_file, ensure_ascii=False, indent=4)

with open(anno_root + '/test.json', 'w') as test_file:
    json.dump(test, test_file, ensure_ascii=False, indent=4)

with open(anno_root + '/val.json', 'w') as val_file:
    json.dump(val, val_file, ensure_ascii=False, indent=4)