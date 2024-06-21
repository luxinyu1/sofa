import json
import argparse
from inferencer import OpenAIInferencer
from fastchat.conversation import get_conv_template
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--template", type=str, required=True)
parser.add_argument("--api-base", type=str)
parser.add_argument("--api-key", type=str, required=True)
parser.add_argument("--eval-file", type=str, default="./data/scope_probe.json")
parser.add_argument("--judge-with-position", action="store_true")
parser.add_argument("--unrelated-instructions", type=str, default="./data/alpaca_data_gpt4.json")
parser.add_argument("--prepend-system-message-as-user", action="store_true")
parser.add_argument("--output-dir", default="./outputs/")
parser.add_argument("--k-th-turn", type=int, default=1)
parser.add_argument("--output-file-name", type=str, default="scope.json")
parser.add_argument("--eval-output-file", action="store_true")
args = parser.parse_args()

all_test_inputs = []
all_conflict_num = 0
all_similar_num = 0
all_regular_num = 0
with open(args.eval_file, "r", encoding="utf-8") as f:
    all_test_points = json.loads(f.read())
    for d in all_test_points:
        processed_data_point = dict()
        processed_data_point["system"] = d["rule"]
        processed_data_point["input"] = d["instruction_in_scope"]
        processed_data_point["in_scope"] = True
        processed_data_point["detect_string"] = d["detect_string"]
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
        if d["conflict"] == "False" and d["similar"] == "False":
            all_regular_num += 1

    del all_test_points

print(all_conflict_num)
print(all_similar_num)

def string_matching(res, matching_string, position):
    res = res.lower()
    matching_string = matching_string.lower()
    if position == "left":
        return res.startswith(matching_string)
    elif position == "right":
        return res.endswith(matching_string)
    elif position == "whole":
        return matching_string in res
    else:
        raise NotImplementedError

# print(all_test_inputs[:2])

if not args.eval_output_file:

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
            else: # the unrelated turn
                conv = all_convs[conv_idx]
                unrelated_utterance = all_unrelated_instructions[unrelated_idx] + " "
                if i == 2 and args.prepend_system_message_as_user:
                    conv.append_message(conv.roles[0], probe["system"] + " " + unrelated_utterance)
                else:
                    conv.append_message(conv.roles[0], unrelated_utterance)
                unrelated_idx += 1

            conv_idx += 1

        turn_inputs = [conv.to_openai_api_messages() for conv in all_convs]
        if args.prepend_system_message_as_user:
            turn_inputs = [inputs[1:] for inputs in turn_inputs]
            # remove the first system message
        # print(turn_inputs[0])
        
        all_responses = inferencer.multiprocess_chat_inference(turn_inputs, num_workers=8)
        all_responses = [r.lstrip() for r in all_responses]
        for conv, res in zip(all_convs, all_responses):
            conv.append_message(conv.roles[1], res)

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

else:
    
    output_file_path = os.path.join(args.output_dir, args.model_name, args.output_file_name)
    all_responses = []
    with open(output_file_path, "r", encoding="utf-8") as f:
        for line in f:
            all_responses.append(json.loads(line)["response"])

related_match = 0
unrelated_match = 0
similar_unmatch = 0
case_pass = 0
conflict_match = 0
conflict_case_pass = 0
similar_case_pass = 0
regular_case_pass = 0

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
            print("In scope mismatch:")
            print(i)
            print(r)
            print("============")
    else:
        if string_matching(r, i["detect_string"], match_side):
            unrelated_match += 1
            print("Out of scope match:")
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
                if i["conflict"] == "False" and i["similar"] == "False":
                    regular_case_pass += 1

all_scores = {}

print(related_match)
print(len(all_test_inputs) / 2)

all_scores["related_match"] = related_match / (len(all_test_inputs) / 2)
all_scores["unrelated_match"] = unrelated_match / (len(all_test_inputs) / 2)
all_scores["similar_unmatch"] = similar_unmatch / all_similar_num
all_scores["conflict_match"] = conflict_match / all_conflict_num
all_scores["similar_case_pass"] = similar_case_pass / all_similar_num
all_scores["conflict_case_pass"] = conflict_case_pass / all_conflict_num
all_scores["regular_case_pass"] = regular_case_pass / all_regular_num
all_scores["case_pass"] = case_pass / (len(all_test_inputs) / 2)

print(all_scores)

output_score_file_path = os.path.join(args.output_dir, args.model_name, "score_" + args.output_file_name)
with open(output_score_file_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(all_scores, ensure_ascii=False))