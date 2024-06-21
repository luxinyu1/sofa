import torch
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LogitsProcessorList,
    pipeline
)
import time
from torch.utils.data import DataLoader
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed
)
from typing import List, Dict, Union
import os
from typing import List
from fastchat.model import load_model
from vllm import LLM, SamplingParams
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
import tiktoken

class OpenAIInferencer:

    def __init__(
        self,
        model:str,
        api_key:str,
        api_base:str=None,
        temperature:float=0,
        max_tokens:int=1024,
        request_timeout=120,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        openai.api_key = api_key
        if api_base is not None:
            openai.api_base = api_base

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                        [wait_fixed(5) for i in range(2)] +
                        [wait_fixed(10)]))
    def inference(self, prompt, stop=None):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop
        )
        return response['choices'][0]['text'].strip()

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                        [wait_fixed(5) for i in range(2)] +
                        [wait_fixed(10)]))
    def chat_inference(self, messages: List[Dict]):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                request_timeout=self.request_timeout,
            )
        except Exception as e:
            if str(e).startswith("Detected an error in the prompt. Please try again with a different prompt.") or \
                str(e).startswith("Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt."):
                print(e)
                return "Prompt Error"
            print(e)
        return response['choices'][0]['message']['content'].strip()
    
    def multiprocess_chat_inference(self, 
                                    batch_of_messages: List[List[Dict]], 
                                    num_workers:int=8):

        pool = ThreadPool(num_workers)
        all_completions = list(
            tqdm(pool.imap(self.chat_inference, batch_of_messages), 
                 total=len(batch_of_messages))
        )

        return all_completions

    def encode(self, prompt:str) -> List:
        enc = tiktoken.encoding_for_model(self.model)
        return enc.encode(prompt)
    
    def decode(self, tokens:List) -> str:
        enc = tiktoken.encoding_for_model(self.model)
        return enc.decode(tokens)

class MultiProcessHFPipelineInferencer:

    def __init__(self, 
                model_path:str, 
                do_sample: bool=False,
                num_gpus:int=None, 
                num_beams:int=1,
                top_p:int=1,
                max_new_tokens:int=512,
                temperature:float=0
                ):

        self.num_gpus = num_gpus
        
        self.manager = mp.Manager()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.model.generation_config = GenerationConfig(
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

    def inference_on_device(self, device, data, shared_dict, batch_size=32):

        self.model.to(device)
        
        pipe = pipeline(
            task="text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=device.index, 
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )

        for i in tqdm(range(0, len(data), batch_size)):
            batched_inp = []
            for j in data[i: i + batch_size]:
                batched_inp.append(j)

            output = pipe(batched_inp, return_full_text=False)
                
            for k in range(len(batched_inp)):
                shared_dict[(i + k) * self.num_gpus + device.index] = output[k][0]["generated_text"]

    def inference(self, data:List, batch_size=32):

        processes = []

        shared_dict = self.manager.dict()

        for device_id in range(self.num_gpus):
            device = torch.device(f"cuda:{device_id}")
            data_on_device = data[device_id::self.num_gpus]
            p = mp.Process(
                target=self.inference_on_device, 
                args=(device, data_on_device, shared_dict, batch_size)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        sorted_results = [value for key, value in sorted(shared_dict.items(), key=lambda item: item[0])]

        return sorted_results

    def get_attn_on_device(self, device, data, shared_dict, head_reduce):

        self.model.to(device)
        self.model.eval()
        for i in tqdm(range(0, len(data))):

            inputs = self.tokenizer(data[i], return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            with torch.no_grad():
                outputs = self.model(input_ids, output_attentions=True)
                attns = outputs.attentions # Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
                attns = list(attns)
                for i in range(len(attns)):
                    attns[i] = attns[i].cpu().numpy()
                    if head_reduce == "mean":
                        attns[i] = attns[i].mean(axis=1)
                    elif head_reduce is None:
                        pass
                    else:
                        raise NotImplementedError()

                shared_dict[i*self.num_gpus+device.index] = attns
                del attns
    
    def get_attn(self, data:List, head_reduce=None):

        processes = []

        shared_dict = self.manager.dict()
        for device_id in range(self.num_gpus):
            device = torch.device(f"cuda:{device_id}")
            data_on_device = data[device_id::self.num_gpus]
            p = mp.Process(
                target=self.get_attn_on_device, 
                args=(device, data_on_device, shared_dict, head_reduce)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        sorted_results = [value for key, value in sorted(shared_dict.items(), key=lambda item: item[0])]

        return sorted_results


    def get_probs_on_device(self, device, data, shared_dict):

        self.model.to(device)
        self.model.eval()
        for i in tqdm(range(0, len(data))):

            inputs = self.tokenizer(data[i], return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            # print(input_ids.shape)
            attention_mask = inputs.attention_mask.to(device)
            with torch.no_grad():
                outputs = self.model(input_ids)[0].squeeze(0)
            log_probs = outputs.log_softmax(-1).cpu().numpy()
            # print(log_probs.shape)
            log_probs = log_probs[:-1,:] # deal with the shift
            # print(log_probs.shape)
            log_probs = log_probs[range(log_probs.shape[0]), inputs.input_ids.squeeze(0)[1:]]
            # print(log_probs.shape)

            shared_dict[i*self.num_gpus + device.index] = log_probs

    def get_probs(self, data:List):

        processes = []

        shared_dict = self.manager.dict()

        for device_id in range(self.num_gpus):
            device = torch.device(f"cuda:{device_id}")
            data_on_device = data[device_id::self.num_gpus]
            p = mp.Process(
                target=self.get_probs_on_device, 
                args=(device, data_on_device, shared_dict)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        sorted_results = [value for key, value in sorted(shared_dict.items(), key=lambda item: item[0])]

        return sorted_results