from vllm import LLM, SamplingParams
from typing import List
import multiprocessing as mp
import os
from tqdm import tqdm

class MultiProcessVllmInferencer:

    def __init__(self, 
                model_path:str, 
                num_gpus:int,
                do_sample:bool=False,
                num_beams:int=1,
                max_new_tokens:int=512,
                temperature:float=0,
                top_p:float=1.0,
                top_k:int=-1,
                frequency_penalty=0.0,
                model_max_tokens=4096,
                ):

        self.num_gpus = num_gpus

        self.manager = mp.Manager()

        self.model_path = model_path
        self.model_max_tokens = model_max_tokens

        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            use_beam_search=(not do_sample) and (not num_beams==1),
            best_of=num_beams,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            length_penalty=1.0,
            frequency_penalty=frequency_penalty,
            early_stopping=False,
        )

    def inference_on_device(self, device_id, data, shared_dict, batch_size=32):

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id) # hacking the visible devices to isolate gpus

        model = LLM(
            model=self.model_path, # for small models, tensor parallel doesn't accelerate much
            tensor_parallel_size=1,
            max_num_batched_tokens=self.model_max_tokens,
        )

        for i in tqdm(range(0, len(data), batch_size)):
            batched_inp = []
            for j in data[i: i + batch_size]:
                batched_inp.append(j)

            outputs = model.generate(
                batched_inp, 
                self.sampling_params,
                use_tqdm=False
            )
                
            for k in range(len(batched_inp)):
                shared_dict[(i + k) * self.num_gpus + device_id] = outputs[k].outputs[0].text

    def inference(self, data:List, batch_size=32):

        processes = []
        shared_dict = self.manager.dict()

        for device_id in range(self.num_gpus):
            data_on_device = data[device_id::self.num_gpus]
            p = mp.Process(
                target=self.inference_on_device, 
                args=(device_id, data_on_device, shared_dict, batch_size)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        sorted_results = [value for key, value in sorted(shared_dict.items(), key=lambda item: item[0])]

        return sorted_results