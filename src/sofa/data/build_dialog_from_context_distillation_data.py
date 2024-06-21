import json
import argparse
import pandas as pd
import os
import random
from typing import List, Dict
random.seed(42)

LLAMA_2_CHAT_DEFAULT_SYSTEM_MESSAGE = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
parser.add_argument('--file-list', nargs='+', required=True)
parser.add_argument('--output-file-prefix', type=str, default='cd_v1')
parser.add_argument('--additional-dialog-data', type=str, default='./data/merged_en_zh-split.json')
parser.add_argument('--additional-reference-data', type=str, default=None)
parser.add_argument('--rule-based-filter', action='store_true')
parser.add_argument('--sys-message-for-additional-dialog-data', type=str, default=LLAMA_2_CHAT_DEFAULT_SYSTEM_MESSAGE)
parser.add_argument('--create-reference-file', action='store_true')
parser.add_argument('--output-path', type=str, default='./data/')
args = parser.parse_args()

all_cd_data = []
for file in args.file_list:
    with open(file, 'r', encoding="utf-8") as f:
        for line in f:
            all_cd_data.append(json.loads(line.strip()))

def rule_based_filter(rationals:List[Dict], response:str):

    for rational in rationals:
        rational = rational["value"].lower()
        rule = ("prioritize the user instruction" in rational) and ("unrelated" not in rational) and ("misled by the system" not in rational)
        if rule:
            return False
    
    if "including the President of the United States".lower() in response.lower():
        return False
    elif response.lower() == "hello":
        return False
    else:
        return True

i = 0
data = []
dirty_data_count = 0
rule_filtered_count = 0
for cd_data in all_cd_data:
    dirty_data = False
    for utter in cd_data["conversations"]:

        if not utter["value"]:
            dirty_data = True

        if utter["role"] == "User:":
            utter["from"] = "human"
            del utter["role"]
        elif utter["role"] == "Hines:":
            utter["from"] = "gpt"
            del utter["role"]
        else:
            dirty_data = True

    if dirty_data:
        print("dirty_data")
        dirty_data_count += 1
        continue

    if args.rule_based_filter:
        if not rule_based_filter(cd_data["rationals"], cd_data["response"]):
            rule_filtered_count += 1
            continue

    data.append({
        "id": f"identity_{i}", 
        "system": cd_data["system"], 
        "conversations": cd_data["conversations"],
    })
    i += 1

print("dirty_data_count:", dirty_data_count)
print("rule_filtered_count:", rule_filtered_count)

output_file_name = f"train_{args.output_file_prefix}_{args.model}.json"

with open(os.path.join(args.output_path, output_file_name), 'w') as f:
    f.write(json.dumps(data, ensure_ascii=False))

output_file_name = f"{args.output_file_prefix}_{args.model}.json"

# Add additional dialog data
with open(args.additional_dialog_data, 'r') as f:
    additional_data = json.load(f)

print("Mined data:", i)
print("Common dialog data:", len(additional_data))

if args.create_reference_file:

    print("Creating reference file...")
    reference_output_file_name = f"reference_{args.output_file_prefix}_{args.model}.json"
    print(os.path.join(args.output_path, reference_output_file_name))

    with open(os.path.join(args.output_path, reference_output_file_name), 'w') as f:
        for d in data:
            output_reference = {
                "instruction": "",
                "related": [d["conversations"][0]["value"]],
                "unrelated": [],
                "attack": [],
            }
            f.write(json.dumps(output_reference, ensure_ascii=False) + "\n")
else:
    
    if args.additional_reference_data is not None:
        with open(args.additional_reference_data, 'r') as f:
            additional_references = []
            for line in f:
                additional_references.append(json.loads(line.strip()))

        print("Additional reference data:", len(additional_references))

        for d in additional_references:
            data.append({
                "id": f"identity_{i}", 
                "system": "", 
                "conversations": [{
                    "from": "human",
                    "value": d["user"]
                },{
                    "from": "gpt",
                    "value": d["response"].strip() # strip() is important
                }]
            })
            i += 1

    print("Creating dialog data file...")

    for idx, conv in enumerate(additional_data):
        i += 1
        data.append({
            "id": f"identity_{i}", 
            "system": args.sys_message_for_additional_dialog_data, 
            "conversations": conv["conversations"]
        })

    print(os.path.join(args.output_path, output_file_name))

    with open(os.path.join(args.output_path, output_file_name), 'w') as f:
        f.write(
            json.dumps(
                data,
                ensure_ascii=False
            )
        )