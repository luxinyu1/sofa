"""Script to evaluate test cases."""
import argparse
from collections import defaultdict
import concurrent.futures
from dataclasses import asdict
import json
import os
from typing import List

from llm_rules import Role, Message, models, scenarios

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="data/systematic")
    parser.add_argument("--output_dir", type=str, default="logs/systematic")
    parser.add_argument("--output_tags", type=str, default="", help="Tags to add to output directory name")
    parser.add_argument(
        "--provider", type=str, default="openai", choices=models.PROVIDER_NAMES
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0613",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Evaluate a single scenario, or all scenarios if None",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Evaluate a single category, or all categories if None",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="If a single scenario is specified, evaluate a single test case by ID",
    )
    parser.add_argument(
        "--system_instructions",
        action="store_true",
        default=False,
        help="Present instructions as a system message, if supported",
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default=None,
        choices=models.SYSTEM_MESSAGES.keys(),
        help="System message to model, if not using --system_instructions",
    )
    parser.add_argument(
        "--prefix_message",
        type=str,
        default=None,
        choices=models.SYSTEM_MESSAGES.keys(),
        help="Prefix message to instruction prompt",
    )
    parser.add_argument(
        "--suffix_dir",
        type=str,
        default=None,
        help="Directory containing GCG attack output logs",
    )
    parser.add_argument(
        "--manual_suffix",
        action="store_true",
        default=False,
        help="Prompt user to manually specify adversarial suffix",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max number of concurrent API calls",
    )
    return parser.parse_args()


class AccuracyMeter:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, result):
        self.correct += int(result)
        self.total += 1

    @property
    def accuracy(self):
        return self.correct / self.total if self.total else 0


def load_dataset(args: argparse.Namespace):
    dataset = defaultdict(dict)
    files = sorted(os.listdir(args.test_dir))

    if args.scenario:
        files = [f for f in files if f.startswith(args.scenario)]

    for file in files:
        print("Loading: {}".format(file))
        scenario_name = os.path.splitext(file)[0]
        behavior_name = ""
        if "_" in scenario_name:
            scenario_name, behavior_name = scenario_name.split("_")

        with open(os.path.join(args.test_dir, file)) as f:
            testcases = [json.loads(line) for line in f.readlines()]

            for t in testcases:
                if "category" not in t:
                    t["category"] = "default"
                if "id" not in t:
                    t["id"] = None

            if args.category:
                print("\tFiltering by category: {}".format(args.category))
                testcases = [
                    t
                    for t in testcases
                    if t["category"].lower() == args.category.lower()
                ]

            if args.id:
                print("\tFiltering by id: {}".format(args.id))
                testcases = [t for t in testcases if t["id"] == args.id]

            if len(testcases) == 0:
                print("\tNo testcases remaining, skipped")
                continue
            dataset[scenario_name][behavior_name] = testcases

    return dataset


def evaluate_single(
    scenario: scenarios.BaseScenario,
    user_messages: List[Message],
    model: models.BaseModel,
    use_system_instructions: bool,
    system_message: str,
    prefix_message: str,
    suffix: str,
):
    full_prompt = scenario.prompt

    if prefix_message:
        full_prompt = models.SYSTEM_MESSAGES[prefix_message] + "\n" + full_prompt

    if use_system_instructions:
        messages = [Message(Role.SYSTEM, full_prompt)]
    else:
        system_message = models.SYSTEM_MESSAGES[args.system_message]
        messages = [
            Message(Role.SYSTEM, system_message),
            Message(Role.USER, full_prompt),
            Message(Role.ASSISTANT, scenario.initial_response),
        ]

    for i, m in enumerate(user_messages):
        if i == 0 and suffix:
            m.content += " " + suffix  # gcg suffixes are applied with a preceding space
        messages.append(m)

        response = models.call_with_retries(model, messages)
        content = models.concat_stream(response)
        messages.append(Message(Role.ASSISTANT, content))

        result = scenario.evaluate(messages, use_system_instructions)

        if not result.passed:
            print("x", end="", flush=True)
            return result, messages

    print(".", end="", flush=True)
    return result, messages


def main(args):
    model_name = args.model.split("/")[-1]
    print("model_name:", model_name)
    suite_name = args.test_dir.split("/")[-1]
    print("suite_name:", suite_name)

    args.output_dir = os.path.join(args.output_dir, model_name + args.output_tags)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.provider == "transformers" or args.provider == "vllm":
        args.concurrency = 1

    model = models.MODEL_BUILDERS[args.provider](
        model=args.model,
        temperature=0,
    )

    dataset = load_dataset(args)
    all_categories = set()

    print("\n=== Beginning evaluation ===")

    results = defaultdict(AccuracyMeter)
    results_by_category = defaultdict(AccuracyMeter)

    pass_num = 0
    total_num = 0

    for name in scenarios.SCENARIOS:
        print(name)
        if name not in dataset:
            print("Skip")
            continue

        for subset in dataset[name]:
            fullname = f"{name}_{subset}" if subset else name
            print(f"{fullname}: ", end="")

            output_file = os.path.join(args.output_dir, f"{suite_name}_{fullname}.jsonl")
            if os.path.exists(output_file):
                os.remove(output_file)

            suffix = ""
            if args.suffix_dir:
                suffix_file = os.path.join(args.suffix_dir, f"{fullname}.jsonl")
                if not os.path.exists(suffix_file):
                    print(f"Suffix file for {fullname} not found, skipping scenario")
                    continue
                with open(suffix_file) as f:
                    suffix = json.loads(f.readlines()[-1].strip())["suffix"]
            elif args.manual_suffix:
                suffix = input("Enter suffix: ")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as executor:
                thread_to_idx = {}
                for i, sample in enumerate(dataset[name][subset]):
                    scenario = scenarios.SCENARIOS[name](sample["params"])
                    messages = Message.unserialize(sample["messages"])
                    thread = executor.submit(
                        evaluate_single,
                        scenario,
                        messages,
                        model,
                        args.system_instructions and model.supports_system_message,
                        args.system_message,
                        args.prefix_message,
                        suffix,
                    )
                    thread_to_idx[thread] = i
                    all_categories.add(sample["category"])

                for t in concurrent.futures.as_completed(thread_to_idx):
                    result, messages = t.result()
                    if result.passed:
                        pass_num += 1
                        total_num += 1
                    else:
                        total_num += 1
                    sample = dataset[name][subset][thread_to_idx[t]]
                    category = sample["category"]
                    results[f"{fullname}"].update(result.passed)
                    results_by_category[f"{fullname}_{category}"].update(result.passed)
                    # save outputs as jsonl
                    with open(output_file, "a") as f:
                        sample_ = sample.copy()
                        sample_["messages"] = Message.serialize(messages)
                        sample_["result"] = asdict(result)
                        f.write(json.dumps(sample_, sort_keys=True) + "\n")

            rate = 100 * results[f"{fullname}"].accuracy
            print(f"\t{rate:.1f}%")

    # Print results in copy-pastable format: for each scenario, print average then all categories
    all_categories = sorted(list(all_categories))
    result_str = "name,Average," + ",".join(all_categories) + "\n"
    for name in results:
        acc = 100 * results[name].accuracy
        result_str += f"{name},{acc:.3f}"
        for category in all_categories:
            name_cat = f"{name}_{category}"
            if name_cat in results_by_category:
                acc = 100 * results_by_category[name_cat].accuracy
                result_str += f",{acc:.3f}"
            else:
                result_str += ",-"
        result_str += "\n"

    print("\ncopypaste:")
    output_file = os.path.join(args.output_dir, f"score_{suite_name}_{args.scenario}.csv")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result_str+"\n")
        f.write(str(pass_num)+"\n")
        f.write(str(total_num))
    print(result_str)

if __name__ == "__main__":
    args = parse_args()
    main(args)