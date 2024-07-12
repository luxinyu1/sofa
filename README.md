This repository contains the code and datasets used in the research paper titled "SoFA: Shielded On-the-fly Alignment via Priority Rule Following".

## Install

To set up the necessary environment, run the following commands:

```bash
conda create -n sofa python=3.10
conda activate sofa
pip install -e .
```

## Data Paraparation

Download the necessary data using the provided script:

```bash
bash ./scripts/download.sh
```
| File Name | Discription |
| ------- | ------- |
| ```data/merged_en_zh-split.json``` | ShareGPT Data |
| ```data/merged_en_zh-split_uncensored.json``` | Uncensored ShareGPT Data |
| ```data/cd_v5_wo_sys_align_gpt-3.5-turbo-1106.json``` | ShareGPT Data + PriorityRules |
| ```data/cd_v5_uncensored_wo_sys_align_gpt-3.5-turbo-1106.json``` | Uncensored ShareGPT Data + PriorityRules |

## Train
Fine-tune models using the corresponding scripts located in the scripts directory: ```scripts/finetune_${model_name}_${data}.sh```.

## Evaluation

Evaluate the models use the ```./scripts/eval_${benchmark_name}.sh``` under the scripts dir.

For few-shot experiments, we recommend setting up a new environment and installing the `lm-evaluation-harness` test suite:

```
conda create -n lm_eval python=3.10
conda activate lm_eval
wget --no-check-certificate https://github.com/EleutherAI/lm-evaluation-harness/archive/refs/tags/v0.4.0.zip
unzip v0.4.0.zip && mv lm-evaluation-harness-0.4.0/ lm-evaluation-harness/
cd lm-evaluation-harness
pip install -e .
```

## Citation

Please cite our work if you find our code and dataset useful:

```
@article{lu2024sofa,
  title={SoFA: Shielded On-the-fly Alignment via Priority Rule Following},
  author={Lu, Xinyu and Yu, Bowen and Lu, Yaojie and Lin, Hongyu and Yu, Haiyang and Sun, Le and Han, Xianpei and Li, Yongbin},
  journal={arXiv preprint arXiv:2402.17358},
  year={2024}
}
```
