import argparse
import json
from tqdm import tqdm
import os
from fastchat.conversation import get_conv_template

from vllm_inferencer import MultiProcessVllmInferencer

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--template", type=str, required=True)
parser.add_argument("--k-th-turn", type=int, default=1)
parser.add_argument("--probe-file", type=str, required=True)
parser.add_argument("--unrelated_instructions", type=str, default="./data/alpaca_data_gpt4.json")
parser.add_argument("--test-aspect", choices=["related", "unrelated", "attack", "all"], required=True)
parser.add_argument("--max-new-tokens-per-utterance", type=int, default=1024)
parser.add_argument("--output-dir", default="./outputs/")
parser.add_argument("--num-gpus", type=int, default=4)
args = parser.parse_args()

all_test_points = []

inferencer = MultiProcessVllmInferencer(
    args.model_path,
    num_gpus=args.num_gpus,
    max_new_tokens=args.max_new_tokens_per_utterance,
)

all_unrelated_instructions = []
with open(args.unrelated_instructions, "r", encoding="utf-8") as f:
    res = json.loads(f.read())
    for d in res:
        if d["input"]:
            all_unrelated_instructions.append(d["instruction"]+"\n"+d["input"])
        else:
            all_unrelated_instructions.append(d["instruction"])

probe_file_name = "_".join([
    "".join(os.path.basename(args.probe_file).split(".")[:-1]), 
    args.test_aspect, 
    str(args.k_th_turn),
    args.model_name
]) + ".json"

os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
output_file_path = os.path.join(args.output_dir, args.model_name, probe_file_name)
print(output_file_path)

unrelated_idx = 0
conv_idx = 0
all_convs = []

for i in range(args.k_th_turn): # n-1 unrelated instruction -> sample answer
    
    for d in all_test_points:
        if args.test_aspect == "related":
            probes = d["related"]
        elif args.test_aspect == "unrelated":
            probes = d["unrelated"]
        elif args.test_aspect == "attack":
            probes = d["attack"]
        elif args.test_aspect == "all":
            probes = d["related"] + d["unrelated"] + d["attack"]
        for probe in probes:
            if i == 0:
                conv = get_conv_template(args.template)
                conv.system_message = d["instruction"]
                all_convs.append(conv)
            if i == args.k_th_turn - 1: # the last turn
                conv = all_convs[conv_idx]
                assert type(probe) == str
                conv.append_message(conv.roles[0], probe)
                conv.append_message(conv.roles[1], None)
            else: # the unrelated turn
                conv = all_convs[conv_idx]
                unrelated_utterance = all_unrelated_instructions[unrelated_idx] + " " # Note we should have an additional space here for llama
                conv.append_message(conv.roles[0], unrelated_utterance)
                unrelated_idx += 1
            conv_idx += 1

    turn_inputs = [conv.get_prompt() for conv in all_convs]

    print("===============")
    print(turn_inputs[0])
    print("===============")
        
    all_responses = inferencer.inference(turn_inputs)
    all_responses = [r.lstrip() for r in all_responses]
    for conv, res in zip(all_convs, all_responses):
        conv.append_message(conv.roles[1], res)

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