import argparse
import json
from tqdm import tqdm
import os
from datasets import load_dataset
import multiprocessing as mp
from fastchat.conversation import get_conv_template # bypass fastchat import acclerate
import matplotlib.pyplot as plt
import string

from transformers import AutoTokenizer
from vllm_inferencer import MultiProcessVllmInferencer

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--template", type=str, required=True)
parser.add_argument("--k-th-turn", type=int, default=1)
parser.add_argument("--dataset-format", type=str, default="json")
parser.add_argument("--data-dir", type=str, default=None)
parser.add_argument("--test-data-file", type=str, default=None)
parser.add_argument("--system-message", default=None, type=str)
parser.add_argument("--unrelated-instructions", type=str, default="./data/alpaca_data_gpt4.json")
parser.add_argument("--output-file-name", type=str, default="truthfulqa_mc.json")
parser.add_argument("--max-new-tokens-per-utterance", type=int, default=1024)
parser.add_argument("--output-dir", default="./outputs/")
parser.add_argument("--num-gpus", type=int, default=4)
parser.add_argument("--choice-trigger-prompt", type=str, default="The answer is")
parser.add_argument("--visualize-attn", action="store_true")
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

all_test_points = []
for d in test_dataset:
    all_test_points.append(d["input"])

all_test_points = all_test_points

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

os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
output_file_path = os.path.join(args.output_dir, args.model_name, args.output_file_name)
print(output_file_path)

unrelated_idx = 0
all_convs = []
# build conversation

for i in range(args.k_th_turn): # n-1 unrelated instruction -> sample answer

    conv_idx = 0
    
    for probe in all_test_points:
        
        if i == 0:
            conv = get_conv_template(args.template)
            conv.system_message = args.system_message
            all_convs.append(conv)
        if i == args.k_th_turn - 1: # the last turn
            conv = all_convs[conv_idx]
            conv.append_message(conv.roles[0], probe)
            conv.append_message(conv.roles[1], None)
        else: # the unrelated turn
            conv = all_convs[conv_idx]
            unrelated_utterance = all_unrelated_instructions[unrelated_idx]
            conv.append_message(conv.roles[0], unrelated_utterance)
            conv.append_message(conv.roles[1], None)
            unrelated_idx += 1

        conv_idx += 1

    if i == args.k_th_turn - 1:
        turn_inputs = [conv.get_prompt() + " " + args.choice_trigger_prompt for conv in all_convs]
    else:
        turn_inputs = [conv.get_prompt() for conv in all_convs]
    
    print(turn_inputs[0])
    all_responses = inferencer.inference(turn_inputs)
    all_responses = [r.lstrip() for r in all_responses]
    for conv, res in zip(all_convs, all_responses):
        conv.update_last_message(res)

if args.visualize_attn:

    all_final_texts = [conv.get_prompt() for conv in all_convs]
    all_attn = inferencer.get_attn(all_final_texts, head_reduce="mean")

    # Calculate the average attention map

    all_layer_attention_add_res = []

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ## get system messages
    conv = get_conv_template(args.template)
    conv.system_message = args.system_message
    system_prompt = conv.get_prompt()
    len_input_ids = len(tokenizer(system_prompt).input_ids)
    all_tokens = tokenizer.convert_ids_to_tokens(tokenizer(system_prompt).input_ids)

    for (i, attn_res) in enumerate(all_attn): # instance loop
        if i == 0:
            for layer_res in attn_res:
                layer_res = layer_res[:,:len_input_ids,:len_input_ids]
                layer_res = layer_res.squeeze(axis=0) # squeeze the batch dim
                all_layer_attention_add_res.append(layer_res)
        else:
            for (j, layer_res) in enumerate(attn_res):
                layer_res = layer_res[:,:len_input_ids,:len_input_ids]
                layer_res = layer_res.squeeze(axis=0) # squeeze the batch dim
                all_layer_attention_add_res[j] += layer_res

    for j in range(len(all_layer_attention_add_res)):
        all_layer_attention_add_res[j] /= len(all_attn)
        plt.figure(figsize=(30, 30))
        plt.matshow(all_layer_attention_add_res[j], fignum=1)
        plt.colorbar()
        plt.title(f"Layer {j} Average Attention")
        plt.xlabel("Tokens in Sequence")
        plt.ylabel("Tokens in Sequence")

        plt.xticks(range(len(all_tokens)), all_tokens, rotation=90)
        plt.yticks(range(len(all_tokens)), all_tokens)

        plt.savefig(os.path.join(args.output_dir, args.model_name, f"truthfulqa_layer_{j}_attention_map.png"))
        plt.close()

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