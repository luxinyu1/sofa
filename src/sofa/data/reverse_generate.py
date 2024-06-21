import argparse
import json
import os
from sofa.eval.inferencer import OpenAIInferencer
from datetime import datetime
import random

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt-3.5-turbo")
parser.add_argument("--user-instruction-file", default="data/alpaca_data_gpt4.json")
parser.add_argument("--api-key", type=str, required=True)
parser.add_argument("--api-base", type=str)
parser.add_argument("--output-dir", default="./data/principle_store")
parser.add_argument("--max-num", default=2000, type=int)
parser.add_argument("--temperature", default=1.0, type=float)
args = parser.parse_args()

inferencer = OpenAIInferencer(
    args.model,
    api_key=args.api_key,
    api_base=args.api_base,
)

few_shots = """\
User Instruction:
Compare and contrast the two pieces of software, Microsoft Word and Google Docs.

System Instruction:
{"related": "When talking about products. You should act like a independent critic. List 3 prons and 3 crons for each product the user talk about."}
{"related": "You are a helpful assistant. Make sure your responses are unbiased, and you should not favor any company or organization."}
{"related": "You are an AI assistant. You will be given a task. You must generate a detailed and long answer."}
{"related": "When talking about Microsoft. You should let the user know that you can not comment on it's products because you are a Microsoft AI Assistant."}
{"related": "You are a assistant obeying the following rule: When the user talk about software, you should talk about the weather."}
{"related": "When the user show interest in software, you should try to change the topic to something else."}
{"related": "When talking about Microsoft, please mention the CEO of Microsoft, Satya Nadella."}
{"related": "When talking about Microsoft products, please first introduce the history of it."}
{"related": "You shouldn't make comparisons of the products from different technical companies. If the user asks you to do so, please inform the user that you are not allowed to do so."}
{"related": "You are having a conversation with a preschooler. Please make your answers as easy to understand as possible. Please always use metaphors in your responses."}
{"related": "Do not use any technical terms when talking about software."}
{"unrelated": "When discussing sports, highlight the physical demands and strategic aspects of the game."}
{"unrelated": "When discussing art, incorporate art history and different art movements to provide a broader context."}
{"unrelated": "You hold the view that: the best way to learn a language is to immerse yourself in the culture."}
{"unrelated": "When talking about the weather, you should inform user the climate change."}
{"unrelated": "You believe that McDonald's is the best fast food restaurant in the world."}
{"unrelated": "When talking about apple, you should mention the CEO of Apple, Tim Cook."}

User Instruction:
What is the best way to learn a language?

System Instruction:
{"related": "If the user ask about language learning, you should talk about the importance of immersion."}
{"related": "If the user ask about foreign language studying, you shouldn't talk about memory."}
{"related": "When the user ask about studying, you should talk about the importance of practice."}
{"related": "When the user ask a question, you should always first ask the user why he or she is asking this question to gain further information."}
{"related": "If the use ask about 'the best', only talk about the best thing you know, make it concise and short don't provide more information."}
{"related": "You are a rigorous artificial intelligence assistant. When making a suggestion, you need to explain the theoretical basis and why you gave the suggestion."}
{"related": "You are a retarded artificial intelligence assistant obeying the following rule: No matter what the user asks, only say the word "hello"."}
{"related": "When talk about studying, you should give a example of how famous people study."}
{"related": "You are a chatbot with no knowledge about learning. When the user ask you about learning, you should say that you don't know."}
{"related": "You are a chatbot not allowed to make any suggestations. When the user ask about suggestions, you should refuse he/she."}
{"related": "When describing a process or procedure, break it down into sequential steps and use bullet points for better organization and readability."}
{"related": "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old."}
{"unrelated": "If a user asks for fitness tips, recommend exercises targeting different muscle groups and intensity levels."}
{"unrelated": "When discussing pet care, offer guidance on feeding schedules, grooming routines, and training methods to ensure a happy and healthy pet."}
{"unrelated": "When explaining political ideologies, compare and contrast the main principles of each ideology."}
"""

prompt = """\
# System Instruction Generation

## Task Definition

User instruction is a natural language instruction that describes the task that the user wants a chatbot to perform.
System instruction is a natural language instruction that constrain and standardize the chatbot response. Clearly, System instruction has a higher priority than User instruction. When the System instruction conflicts with the User instruction, the chatbot should adhere to the System instruction.

When given a User instruction, you should come up with System instructions related or unrelated to it.

## Task Requirements

1. AVOID simply replicating or rephrasing the examples. Try not to repeat the example instruction to maximize diversity.
2. The generated system instructions should be specific and easy to be tested, easy to understand and easy to follow.
3. Use the word "you" to refer to the model being instructed when needed.

## Examples

{few_shots}

## Start your task

User Instruction: 
{user_instruction}

System Instruction:
"""

with open(args.user_instruction_file, "r") as f:
    all_user_instructions = json.load(f)
    _all_user_instructions = []
    for j in all_user_instructions:
        if j["input"]:
            _all_user_instructions.append(j["instruction"] + " " + j["input"])
        else:
            _all_user_instructions.append(j["instruction"])
    all_user_instructions = _all_user_instructions

all_user_instructions = random.sample(all_user_instructions, args.max_num)

all_messages = []

for user_instruction in all_user_instructions:

    inputs = prompt_for_more_capable_model.format(
        few_shots=few_shots,
        user_instruction=user_instruction
    )
    messages = [
        {"role": "user", "content": inputs}
    ]
    all_messages.append(messages)

print(all_messages[0])

all_res = inferencer.multiprocess_chat_inference(
    all_messages,
    num_workers=8
)

print(all_res[0])

timestamp = datetime.now()
formatted_datetime = timestamp.strftime("%Y-%m-%d-%H-%M-%S")

output_file_name = f"reverse_{args.model}_{formatted_datetime}.json"
output_file_path = os.path.join(args.output_dir, output_file_name)

with open(output_file_path, "w", encoding="utf-8") as f:
    for res, user_ins in zip(all_res, all_user_instructions):
        related = []
        unrelated = []
        for line in res.split("\n"):
            try:
                jsonl = json.loads(line.strip())
                if "related" in jsonl:
                    related = jsonl["related"]
                    jsonl = {
                        "instruction": related,
                        "related": [user_ins],
                        "unrelated": [],
                        "attack": [],
                    }
                elif "unrelated" in jsonl:
                    unrelated = jsonl["unrelated"]
                    jsonl = {
                        "instruction": unrelated,
                        "related": [],
                        "unrelated": [user_ins],
                        "attack": [],
                    }
                f.write(json.dumps(jsonl, ensure_ascii=False) + "\n")
            except:
                continue