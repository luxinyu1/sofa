test_model=(
    "Llama-2-7b-chat-hf" "Llama-2-13b-chat-hf"
    "llama-2-7b_sharegpt_wo_sys_align" "llama-2-7b_cd_v5_1106_wo_sys_align" "llama-2-7b_sharegpt_uncensored_wo_sys_align"  "llama-2-7b_cd_v5_1106_uncensored_wo_sys_align" "llama-2-7b_cd_v5_1106_uncensored_wo_sys_align_reference"
    "llama-2-13b_sharegpt_wo_sys_align" "llama-2-13b_cd_v5_1106_wo_sys_align" "llama-2-13b_sharegpt_uncensored_wo_sys_align" "llama-2-13b_cd_v5_1106_uncensored_wo_sys_align" "llama-2-13b_cd_v5_1106_uncensored_wo_sys_align_reference"
)

for MODEL_NAME in "${test_model[@]}"
do

    echo $MODEL_NAME
    python ./src/sofa/eval/calc_rules_score.py --output-dir "./outputs/${MODEL_NAME}/rules/${MODEL_NAME}/"
    echo "==============="

done