export HF_DATASETS_CACHE="./hf_cache/datasets"

MODEL_NAME=""
MODEL_PATH=""

for MODEL_NAME in "${test_model[@]}"
do

    lm_eval --model hf \
        --model_args pretrained="${MODEL_PATH}" \
        --tasks "arc_challenge" \
        --batch_size auto \
        --num_fewshot 25 \
        --output_path "./outputs/${MODEL_NAME}/arc_challenge.json" \

    lm_eval --model hf \
        --model_args pretrained="${MODEL_PATH}" \
        --tasks "hellaswag" \
        --batch_size auto \
        --num_fewshot 10 \
        --output_path "./outputs/${MODEL_NAME}/hellaswag.json" \

    lm_eval --model hf \
        --model_args pretrained="${MODEL_PATH}" \
        --tasks "truthfulqa" \
        --batch_size auto \
        --num_fewshot 0 \
        --output_path "./outputs/${MODEL_NAME}/truthfulqa_mc.json" \

    lm_eval --model hf \
        --model_args pretrained="${MODEL_PATH}" \
        --tasks "gsm8k" \
        --batch_size auto \
        --num_fewshot 5 \
        --output_path "./outputs/${MODEL_NAME}/gsm8k.json" \

    lm_eval --model hf \
        --model_args pretrained="${MODEL_PATH}" \
        --tasks "mmlu" \
        --batch_size auto \
        --num_fewshot 5 \
        --output_path "./outputs/${MODEL_NAME}/mmlu.json"

    lm_eval --model hf \
        --model_args pretrained="${MODEL_PATH}" \
        --tasks "winogrande" \
        --batch_size auto \
        --num_fewshot 5 \
        --output_path "./outputs/${MODEL_NAME}/winogrande.json"

done