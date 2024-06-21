import argparse
import json
from tqdm import tqdm
import os
from datasets import load_dataset
from fastchat.conversation import get_conv_template # bypass fastchat import acclerate

from inferencer import OpenAIInferencer

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--template", type=str, required=True)
parser.add_argument("--k-th-turn", type=int, default=1)
parser.add_argument("--dataset-format", type=str, default="json")
parser.add_argument("--data-dir", type=str, default=None)
parser.add_argument("--test-data-file", type=str, default=None)
parser.add_argument("--api-key", type=str, required=True)
parser.add_argument("--api-base", type=str)
parser.add_argument("--system-message", default=None, type=str)
parser.add_argument("--unrelated-instructions", type=str, default="./data/alpaca_data_gpt4.json")
parser.add_argument("--output-file-name", type=str, default="truthfulqa_mc.json")
parser.add_argument("--max-new-tokens-per-utterance", type=int, default=1024)
parser.add_argument("--output-dir", default="./outputs/")
parser.add_argument("--choice-trigger-prompt", type=str, default="Only output one letter for the choice. The answer is:")
args = parser.parse_args()

if args.test_data_file is not None:
    dataset = load_dataset(
        args.dataset_format,
        data_files={"test": args.test_data_file}
    )
elif args.data_dir is not None:
    dataset = load_dataset(
        args.dataset_format,
        data_dir=args.data_dir,
    )

test_dataset = dataset["test"]

# We only parse the first-turn user prompt from hh-rlhf
all_test_points = []
for d in test_dataset:
    all_test_points.append(d["input"])

inferencer = OpenAIInferencer(
    api_key=args.api_key,
    api_base=args.api_base,
    model=args.model_name,
    temperature=0
)

all_unrelated_instructions = []
with open(args.unrelated_instructions, "r", encoding="utf-8") as f:
    res = json.loads(f.read())
    for d in res:
        if d["input"]:
            all_unrelated_instructions.append(d["instruction"]+"\n"+d["input"])
        else:
            all_unrelated_instructions.append(d["instruction"])

os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
output_file_path = os.path.join(args.output_dir, args.model_name, args.output_file_name)
print(output_file_path)

unrelated_idx = 0
conv_idx = 0
all_convs = []
# build conversation

for i in range(args.k_th_turn): # n-1 unrelated instruction -> sample answer
    
    for probe in all_test_points:
        
        if i == 0:
            conv = get_conv_template(args.template)
            conv.system_message = args.system_message
            all_convs.append(conv)
        if i == args.k_th_turn - 1: # the last turn
            conv = all_convs[conv_idx]
            conv.append_message(conv.roles[0], probe + ". " + args.choice_trigger_prompt)
        else: # the unrelated turn
            conv = all_convs[conv_idx]
            unrelated_utterance = all_unrelated_instructions[unrelated_idx] + " " # Note we should have an additional space here for llama
            conv.append_message(conv.roles[0], unrelated_utterance)
            unrelated_idx += 1

        conv_idx += 1

    print(conv.to_openai_api_messages())

    turn_inputs = [conv.to_openai_api_messages() for conv in all_convs]
    print(turn_inputs[0])
    
    all_responses = inferencer.multiprocess_chat_inference(turn_inputs, num_workers=8)
    all_responses = [r.lstrip() for r in all_responses]
    for conv, res in zip(all_convs, all_responses):
        conv.append_message(conv.roles[1], res)

all_responses = []
with open(output_file_path, "w", encoding="utf-8") as f:
    for conv in all_convs:
        messages = [msg for msg in conv.messages if msg[1] is not None]
        f.write(
            json.dumps(
                {
                    "system": conv.system_template.format(system_message=conv.system_message), 
                    "user": messages[-2][1], 
                    "response": messages[-1][1],
                    "utterences": [m[1] for m in messages[:-2]]
                }, 
                ensure_ascii=False
            )
            +"\n"
        )
        all_responses.append(messages[-1][1])

# print(len(all_responses))
# print(all_responses)
# print("==========================")

acc = 0
error_format = 0

label_mapping = {"a": 0, "b": 1, "c":2, "d": 3}
for res, data_point in zip(all_responses, test_dataset):
    golden = data_point["label"]
    
    if res and (res[0].lower() in label_mapping.keys()):
        res_idx = label_mapping[res[0].lower()]
        if res_idx == golden: 
            acc += 1

    elif res and (res[0] == "(" and res[1].lower() in label_mapping.keys()):
        res_idx = label_mapping[res[1].lower()]
        if res_idx == golden: 
            acc += 1
    else:
        print(res)
        error_format += 1

acc = acc / len(test_dataset)

output_score_filename = "score_" + args.output_file_name
output_score_file_path = os.path.join(args.output_dir, args.model_name, output_score_filename)

scores = {
    "acc": acc,
    "error_format": error_format
}

print(scores)

with open(output_score_file_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(scores))