# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

LLAMA_2_CHAT_DEFAULT_SYSTEM_MESSAGE = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_flash_attn: bool = False
    add_special_tokens: Optional[str] = None


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    template: str = field(default="vicuna")
    new_sys_message: str = field(
        default=None
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    template: str,
    new_sys_message=None,
    sys_messages=None
) -> Dict:
    conv = get_conversation_template(template)

    if conv.sep_style == SeparatorStyle.LLAMA2:
        conv.system_message = LLAMA_2_CHAT_DEFAULT_SYSTEM_MESSAGE # set the empty system_message in fschat==0.28.0

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    dynamic_system_message = False
    if new_sys_message is not None and sys_messages is None:
        rank0_print(f"Replace original system message with {new_sys_message}")
        conv.system_message = new_sys_message
    elif sys_messages is not None and new_sys_message is None:
        assert len(sources) == len(sys_messages)
        dynamic_system_message = True
        rank0_print("Dynamic system message training ...")
    elif sys_messages is not None and new_sys_message is not None:
        raise ValueError()

    # Apply prompt templates
    rank0_print(sources[0]) # [[{"from": "human", "value":}, {"from": "gpt", "value":}, ...]]
    conversations = []
    if dynamic_system_message:
        for i, (source, sys) in enumerate(zip(sources, sys_messages)): # instance loop
            conv.system_message = sys
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source): # dialog loop
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
    else:
        for i, source in enumerate(sources): # instance loop
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source): # dialog loop
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

    rank0_print("Rendered Sample[0]:", conversations[0])

    rank0_print("Max length:", tokenizer.model_max_length)

    # Tokenize all conversations
    input_ids = tokenizer(
        conversations,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids # [dataset_len, max_seq_len]

    targets = input_ids.clone() # [dataset_len, max_seq_len]

    # Mask targets

    first = True
    if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:

        need_adjust:bool = (len(tokenizer.tokenize("</s>USER:")) != len(tokenizer("USER:").input_ids)) # tokenize inconsistancy before transformers==4.33.3

        sep = conv.sep + conv.roles[1] + ": " # " ASSISTANT: " -> "_ASSISTANT:_"
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2) # seperate the encoded conv by the turn end marker "</s>"
            # print(rounds)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID # the very first <s>

            # z = target.clone()
            # z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            # rank0_print(tokenizer.decode(z))
            # rank0_print(tokenizer.convert_ids_to_tokens(z))

            for i, turn in enumerate(turns):
                if turn == "":
                    break

                parts = turn.split(sep)
                # print(parts)
                if len(parts) != 2:
                    rank0_print(
                        f"WARNING: Skipping one utterance with roles > 2."
                    )
                    break
                parts[0] += sep
                turn_len = len(tokenizer(turn).input_ids) # actually we have 1 more token (<s> in the start) but 1 less token (</s> in the end)

                instruction_len = len(tokenizer(parts[0]).input_ids) - 2 # "2" is the <s> in the start and "_" in the end

                if i != 0 and need_adjust:
                    instruction_len -= 1 # when concat with </s> and <s>: USER -> _US ER 
                    turn_len -= 1 # when concat with </s> and <s>: USER -> _US ER

                if False:
                    # rank0_print(tokenizer.convert_ids_to_tokens(tokenizer(turn).input_ids))
                    # rank0_print(tokenizer.convert_ids_to_tokens(tokenizer(parts[0]).input_ids))
                    rank0_print(tokenizer.convert_ids_to_tokens(target[cur_len : cur_len + instruction_len]))

                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len
            
            target[cur_len:] = IGNORE_TOKEN_ID

            if False and first:
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                rank0_print(tokenizer.decode(z))
                rank0_print(tokenizer.convert_ids_to_tokens(z))
                first = False

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    elif conv.sep_style == SeparatorStyle.LLAMA2:
        sep = conv.roles[1] + " " # [/INST]
        for conversation, target in zip(conversations, targets):
            
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2) # " </s><s>"

            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID

            # z = target.clone()
            # z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            # rank0_print(tokenizer.decode(z))
            # rank0_print(tokenizer.convert_ids_to_tokens(z))

            for i, turn in enumerate(turns):
                if turn == "":
                    break
                parts = turn.split(sep)

                if len(parts) != 2:
                    break

                parts[0] += sep
                round_len = len(tokenizer(turn).input_ids) + 2
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if False:
                    rank0_print(tokenizer.convert_ids_to_tokens(target[cur_len : cur_len + instruction_len]))

                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

                cur_len += round_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if first:
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                rank0_print(tokenizer.decode(z))
                rank0_print(tokenizer.convert_ids_to_tokens(z))
                first = False

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                raw_data, 
                tokenizer: transformers.PreTrainedTokenizer,
                template: str,
                new_sys_message=None):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        if "system" in raw_data[0]: 
            sys_messages = [example["system"] for example in raw_data]
            data_dict = preprocess(sources, tokenizer, template, sys_messages=sys_messages)
        else:
            data_dict = preprocess(sources, tokenizer, template, new_sys_message=new_sys_message)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args.data_path, "r"))

    # Split train/test
    np.random.seed(0)
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    if data_args.new_sys_message is not None:
        rank0_print(f"Replacing original system prompt with new: {data_args.new_sys_message}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer, template=data_args.template, new_sys_message=data_args.new_sys_message)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer, template=data_args.template, new_sys_message=data_args.new_sys_message)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if model_args.use_flash_attn:
        rank0_print("Using Flash Attn...")
        from fastchat.train.llama2_flash_attn_monkey_patch import (
            replace_llama_attn_with_flash_attn,
        )
        replace_llama_attn_with_flash_attn()
    
    local_rank = training_args.local_rank
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if model_args.add_special_tokens is not None:
        special_tokens_dict  = {"additional_special_tokens": model_args.add_special_tokens.split(" ")}
        rank0_print("Adding special tokens: {}".format(special_tokens_dict))
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()