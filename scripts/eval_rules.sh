# test the open source models

# test_model=("Llama-2-7b-chat-hf" "Llama-2-13b-chat-hf")

for MODEL_NAME in "${test_model[@]}"
do

    echo ${MODEL_NAME}

    MODEL_PATH="./pretrained_models/${MODEL_NAME}"

    test_scenario=("Encryption" "Integrity" "AccessControl" "Authentication" "Commitment" "Confidentiality" "Millionaires" "DiningCryptographers" "Hodor" "ForbiddenWord" "RockPaperScissors" "AnimalSounds" "Questions" "BinarySearch" "SimonSays");

    # for scenario in "${test_scenario[@]}"
    # do
    #     CUDA_VISIBLE_DEVICES=0 python ./src/sofa/eval/eval_rules.py --test_dir "./data/rules/manual" \
    #                                                                 --scenario ${scenario} \
    #                                                                 --provider "vllm" \
    #                                                                 --model "llama-2@${MODEL_PATH}" \
    #                                                                 --output_dir "./outputs/${MODEL_NAME}/rules/" \
    #                                                                 --system_instructions;
    # done

    for scenario in "${test_scenario[@]}"
    do
        CUDA_VISIBLE_DEVICES=0 python ./src/sofa/eval/eval_rules.py --test_dir "./data/rules/systematic" \
                                                                    --scenario ${scenario} \
                                                                    --provider "vllm" \
                                                                    --model "llama-2@${MODEL_PATH}" \
                                                                    --output_dir "./outputs/${MODEL_NAME}/rules/" \
                                                                    --system_instructions;
    done

done

# test the trained models

# test_model=(
#     "llama-2-7b_sharegpt_wo_sys_align" "llama-2-7b_sharegpt_uncensored_wo_sys_align" "llama-2-7b_cd_v5_1106_wo_sys_align" "llama-2-7b_cd_v5_1106_uncensored_wo_sys_align" "llama-2-7b_cd_v5_1106_uncensored_wo_sys_align_reference"
#     "llama-2-13b_sharegpt_wo_sys_align" "llama-2-13b_sharegpt_uncensored_wo_sys_align" "llama-2-13b_cd_v5_1106_wo_sys_align" "llama-2-13b_cd_v5_1106_uncensored_wo_sys_align" "llama-2-13b_cd_v5_1106_uncensored_wo_sys_align_reference"
# )

test_model=(
    "llama-2-7b_cd_v5_1106_uncensored"
)

for MODEL_NAME in "${test_model[@]}"
do

    echo ${MODEL_NAME}

    MODEL_PATH="./saved_models/${MODEL_NAME}"

    test_scenario=("Encryption" "Integrity" "AccessControl" "Authentication" "Commitment" "Confidentiality" "Millionaires" "DiningCryptographers" "Hodor" "ForbiddenWord" "RockPaperScissors" "AnimalSounds" "Questions" "BinarySearch" "SimonSays");

    for scenario in "${test_scenario[@]}"
    do
        CUDA_VISIBLE_DEVICES=0 python ./src/sofa/eval/eval_rules.py --test_dir "./data/rules/manual" \
                                                                    --scenario ${scenario} \
                                                                    --provider "vllm" \
                                                                    --model "llama-2@${MODEL_PATH}" \
                                                                    --output_dir "./outputs/${MODEL_NAME}/rules/" \
                                                                    --system_instructions;
    done

    for scenario in "${test_scenario[@]}"
    do
        CUDA_VISIBLE_DEVICES=0 python ./src/sofa/eval/eval_rules.py --test_dir "./data/rules/systematic" \
                                                                    --scenario ${scenario} \
                                                                    --provider "vllm" \
                                                                    --model "llama-2@${MODEL_PATH}" \
                                                                    --output_dir "./outputs/${MODEL_NAME}/rules/" \
                                                                    --system_instructions;
    done

done