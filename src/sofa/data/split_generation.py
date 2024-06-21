import random
import json
import argparse
import os
from nltk.tokenize import word_tokenize

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--input-files-list", nargs='+', required=True)
parser.add_argument("--output-file-suffix", type=str, required=True)
parser.add_argument("--output-dir", type=str, default="./data/")
parser.add_argument("--use-unrelated-proportion", type=float, default=1.0)
parser.add_argument("--held-out-test-proportion", type=float, default=0.1)
args = parser.parse_args()

all_data = []
for f in args.input_files_list:
    with open(f, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            data["file"] = f
            all_data.append(data)

def get_statistics(all_data):

    all_statistics = {
        "num_related": 0,
        "num_unrelated": 0,
        "num_attack": 0,
        "num_system": 0,
        "num_user": 0,
        "avg_system_len": 0,
        "avg_user_len": 0,
    }

    unique_system = []
    unique_user = []

    all_system_len = 0
    all_user_len = 0

    for data in all_data:
        all_statistics["num_related"] += len(data["related"])
        all_statistics["num_unrelated"] += len(data["unrelated"])
        all_statistics["num_attack"] += len(data["attack"])

        if data["instruction"] not in unique_system:
            unique_system.append(data["instruction"])
            all_statistics["num_system"] += 1
            all_system_len += len(word_tokenize(data["instruction"]))
        
        all_statistics["num_user"] += (len(data["related"])+len(data["unrelated"])+len(data["attack"]))
        unique_user.extend(data["related"]+data["unrelated"]+data["attack"])

        for ins in data["related"]+data["unrelated"]+data["attack"]:
            all_user_len += len(word_tokenize(ins))

    unique_user = list(set(unique_user))

    all_statistics["num_unique_user"] = len(unique_user)
    all_statistics["avg_system_len"] = all_system_len / all_statistics["num_system"]
    all_statistics["avg_user_len"] = all_user_len / all_statistics["num_user"]

    return all_statistics

all_statistics = get_statistics(all_data)

print(all_statistics)

all_flattened_data = []
for data in all_data:
    for related in data["related"]:
        all_flattened_data.append({
            "instruction": data["instruction"],
            "related": [related],
            "unrelated": [],
            "attack": [],
        })
    for unrelated in data["unrelated"]:
        all_flattened_data.append({
            "instruction": data["instruction"],
            "related": [],
            "unrelated": [unrelated],
            "attack": [],
        })
    for attack in data["attack"]:
        all_flattened_data.append({
            "instruction": data["instruction"],
            "related": [],
            "unrelated": [],
            "attack": [attack],
        })

# Control the unrelated proportion
removed_unrelated = 0
for d in all_flattened_data:
    use_flag = random.random() < args.use_unrelated_proportion
    if len(d["unrelated"]) > 0 and not use_flag:
        removed_unrelated += len(d["unrelated"])
        d["unrelated"] = []
    if len(d["related"]) == 0 and len(d["unrelated"]) == 0 and len(d["attack"]) == 0:
        # remove this data point
        all_flattened_data.remove(d)
        # print("Removed a data point with no related, unrelated, or attack.")

# all_statistics["num_unrelated"] -= removed_unrelated
print("{} unrelated instance removed.".format(removed_unrelated))
# print(all_statistics)

all_statistics = get_statistics(all_flattened_data)
print(all_statistics)

# Split the train and test set
random.shuffle(all_flattened_data)
held_out_test_size = int(len(all_flattened_data)*args.held_out_test_proportion)
train_data = all_flattened_data[held_out_test_size:]
test_data = all_flattened_data[:held_out_test_size]

train_output_file_name = f"train_{args.output_file_suffix}.json"
test_output_file_name = f"test_{args.output_file_suffix}.json"

print(os.path.join(args.output_dir, train_output_file_name))
with open(os.path.join(args.output_dir, train_output_file_name), 'w', encoding="utf-8") as f:
    for d in train_data:
        f.write(json.dumps(d, ensure_ascii=False)+"\n")

print(os.path.join(args.output_dir, test_output_file_name))
with open(os.path.join(args.output_dir, test_output_file_name), 'w', encoding="utf-8") as f:
    for d in test_data:
        f.write(json.dumps(d, ensure_ascii=False)+"\n")