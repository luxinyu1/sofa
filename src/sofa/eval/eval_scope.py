import json
import argparse
from vllm_inferencer import MultiProcessVllmInferencer
from fastchat.conversation import get_conv_template
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--template", type=str, required=True)
parser.add_argument("--eval-file", type=str, default="./data/scope_probe.json")
parser.add_argument("--judge-with-position", action="store_true")
parser.add_argument("--unrelated-instructions", type=str, default="./data/alpaca_data_gpt4.json")
parser.add_argument("--output-dir", default="./outputs/")
parser.add_argument("--k-th-turn", type=int, default=1)
parser.add_argument("--output-file-name", type=str, default="scope.json")
parser.add_argument("--max-new-tokens-per-utterance", type=int, default=1024)
parser.add_argument("--prepend-system-message-as-user", action="store_true")
parser.add_argument("--num-gpus", type=int, default=4)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top-p", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=-1)
args = parser.parse_args()

eval_file = args.eval_file
all_conflict_num = 0
all_similar_num = 0
all_test_inputs = []
with open(args.eval_file, "r", encoding="utf-8") as f:
    all_test_points = json.loads(f.read())
    for d in all_test_points:
        processed_data_point = dict()
        processed_data_point["system"] = d["rule"]
        processed_data_point["input"] = d["instruction_in_scope"]
        processed_data_point["in_scope"] = True
        processed_data_point["detect_string"] =d["detect_string"]
        processed_data_point["detect_position"] = d["position"]
        processed_data_point["conflict"] = d["conflict"]
        processed_data_point["similar"] = d["similar"]
        if d["conflict"] == "True":
            all_conflict_num += 1
        all_test_inputs.append(processed_data_point)
        
        processed_data_point = dict()
        processed_data_point["system"] = d["rule"]
        processed_data_point["input"] = d["instruction_out_scope"]
        processed_data_point["in_scope"] = False
        processed_data_point["detect_string"] = d["detect_string"]
        processed_data_point["detect_position"] = d["position"]
        processed_data_point["conflict"] = d["conflict"]
        processed_data_point["similar"] = d["similar"]
        all_test_inputs.append(processed_data_point)
        if d["similar"] == "True":
            all_similar_num += 1

    del all_test_points

inferencer = MultiProcessVllmInferencer(
    args.model_path,
    num_gpus=args.num_gpus,
    max_new_tokens=args.max_new_tokens_per_utterance,
    top_p=args.top_p,
    top_k=args.top_k,
    temperature=args.temperature,
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

def string_matching(res, matching_string, position):
    res = res.lower()
    matching_string = matching_string.lower()
    if position == "left":
        return res.startswith(matching_string)
    elif position == "right":
        return res.endswith(matching_string)
    elif position == "whole":
        # if matching_string not in res:
            # print(res)
            # print(matching_string)
            # print("===================")
        return matching_string in res
    else:
        raise NotImplementedError

unrelated_idx = 0
conv_idx = 0
all_convs = []
# build conversation

for i in range(args.k_th_turn): # n-1 unrelated instruction -> sample answer
    
    for probe in all_test_inputs:
        
        if i == 0:
            conv = get_conv_template(args.template)
            if args.prepend_system_message_as_user:
                conv.system_message = ""
            else:
                conv.system_message = probe["system"]
            all_convs.append(conv)
        if i == args.k_th_turn - 1: # the last turn
            conv = all_convs[conv_idx]
            if args.k_th_turn == 1 and args.prepend_system_message_as_user:
                conv.append_message(conv.roles[0], probe["system"] + " " + probe["input"])
            else:
                conv.append_message(conv.roles[0], probe["input"])
            conv.append_message(conv.roles[1], None)
        else: # the unrelated turn
            conv = all_convs[conv_idx]
            unrelated_utterance = all_unrelated_instructions[unrelated_idx] + " " # Note we should have an additional space here for llama
            if i == 2 and args.prepend_system_message_as_user:
                conv.append_message(conv.roles[0], probe["system"] + " " + unrelated_utterance)
            else:
                conv.append_message(conv.roles[0], unrelated_utterance)
            unrelated_idx += 1

        conv_idx += 1

    for conv in all_convs:
        # print(conv.to_openai_api_messages())
        pass

    turn_inputs = [conv.get_prompt() for conv in all_convs]
    # print(turn_inputs[0])
    
    all_responses = inferencer.inference(turn_inputs)
    all_responses = [r.lstrip() for r in all_responses]

    assert len(all_convs) == len(all_responses)

    for conv, res in zip(all_convs, all_responses):
        conv.update_last_message(res)

output_file_path = os.path.join(args.output_dir, args.model_name, args.output_file_name)
print(output_file_path)

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

related_match = 0
unrelated_match = 0
similar_unmatch = 0
case_pass = 0
conflict_match = 0
conflict_case_pass = 0
similar_case_pass = 0

print("Failure Mode:")

for r, i in zip(all_responses, all_test_inputs):
    match_side = i["detect_position"] if args.judge_with_position else "whole"
    if i["in_scope"]:
        if string_matching(r, i["detect_string"], match_side):
            related_match += 1
            related_matched = True
            if i["conflict"] == "True":
                conflict_match += 1
        else:
            related_matched = False
            print(i)
            print(r)
            print("============")
    else:
        if string_matching(r, i["detect_string"], match_side):
            unrelated_match += 1
            print(i)
            print(r)
            print("============")
        else:
            if i["similar"] == "True":
                similar_unmatch += 1
            if related_matched:
                case_pass += 1
                if i["conflict"] == "True":
                    conflict_case_pass += 1
                if i["similar"] == "True":
                    similar_case_pass += 1

all_scores = {}

all_scores["related_match"] = related_match / (len(all_test_inputs) / 2)
all_scores["unrelated_match"] = unrelated_match / (len(all_test_inputs) / 2)
all_scores["similar_unmatch"] = similar_unmatch / all_similar_num
all_scores["conflict_match"] = conflict_match / all_conflict_num
all_scores["similar_case_pass"] = similar_case_pass / all_similar_num
all_scores["conflict_case_pass"] = conflict_case_pass / all_conflict_num
all_scores["case_pass"] = case_pass / (len(all_test_inputs) / 2)

output_score_file_path = os.path.join(args.output_dir, args.model_name, "score_" + args.output_file_name)
with open(output_score_file_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(all_scores, ensure_ascii=False))

print(all_scores)

output_score_file_path = os.path.join(args.output_dir, args.model_name, "score_" + args.output_file_name)
with open(output_score_file_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(all_scores, ensure_ascii=False))
