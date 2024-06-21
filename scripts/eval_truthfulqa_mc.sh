# test the open source models

# test_model=("Llama-2-7b-chat-hf" "Llama-2-13b-chat-hf")

for MODEL_NAME in "${test_model[@]}"
do

    echo ${MODEL_NAME}
    MODEL_PATH="./pretrained_models/${MODEL_NAME}"

    python ./src/sofa/eval/eval_truthfulqa_mc.py --dataset-format "arrow" \
                                        --test-data-file "./data/truthful_qa_mc/data-00000-of-00001.arrow" \
                                        --template "llama-2" \
                                        --model-name "${MODEL_NAME}" \
                                        --model-path "${MODEL_PATH}" \
                                        --system-message "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." \
                                        --output-file-name "truthfulqa_mc.json"

    python ./src/sofa/eval/eval_truthfulqa_mc.py --dataset-format "arrow" \
                                        --test-data-file "./data/truthful_qa_mc/data-00000-of-00001.arrow" \
                                        --template "llama-2" \
                                        --model-name "${MODEL_NAME}" \
                                        --model-path "${MODEL_PATH}" \
                                        --system-message "" \
                                        --output-file-name "truthfulqa_mc_wo_sys.json"

done

# test the trained models

# test_model=(
#     "llama-2-7b_sharegpt_wo_sys_align" "llama-2-7b_sharegpt_uncensored_wo_sys_align" "llama-2-7b_cd_v5_1106_wo_sys_align" "llama-2-7b_cd_v5_1106_uncensored_wo_sys_align" "llama-2-7b_cd_v5_1106_uncensored_wo_sys_align_reference"
#     "llama-2-13b_sharegpt_wo_sys_align" "llama-2-13b_sharegpt_uncensored_wo_sys_align" "llama-2-13b_cd_v5_1106_wo_sys_align" "llama-2-13b_cd_v5_1106_uncensored_wo_sys_align" "llama-2-13b_cd_v5_1106_uncensored_wo_sys_align_reference"
# )

# for MODEL_NAME in "${test_model[@]}"
# do

#     echo ${MODEL_NAME}
#     MODEL_PATH="./saved_models/${MODEL_NAME}"

#     python ./src/sofa/eval/eval_truthfulqa_mc.py --dataset-format "arrow" \
#                                         --test-data-file "./data/truthful_qa_mc/data-00000-of-00001.arrow" \
#                                         --template "llama-2" \
#                                         --model-name ${MODEL_NAME} \
#                                         --model-path ${MODEL_PATH} \
#                                         --system-message "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." \
#                                         --output-file-name "truthfulqa_mc.json";

#     python ./src/sofa/eval/eval_truthfulqa_mc.py --dataset-format "arrow" \
#                                         --test-data-file "./data/truthful_qa_mc/data-00000-of-00001.arrow" \
#                                         --template "llama-2" \
#                                         --model-name ${MODEL_NAME} \
#                                         --model-path ${MODEL_PATH} \
#                                         --system-message "" \
#                                         --output-file-name "truthfulqa_mc_wo_sys.json";

# done

# test_model=("Llama-2-7b-chat-hf")

for MODEL_NAME in "${test_model[@]}"
do

    echo ${MODEL_NAME}
    MODEL_PATH="./pretrained_models/${MODEL_NAME}"

    python ./src/sofa/eval/eval_truthfulqa_mc.py --dataset-format "arrow" \
                                        --test-data-file "./data/truthful_qa_mc/data-00000-of-00001.arrow" \
                                        --template "llama-2" \
                                        --model-name "${MODEL_NAME}" \
                                        --model-path "${MODEL_PATH}" \
                                        --system-message "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." \
                                        --k-th-turn 3 \
                                        --output-file-name "truthfulqa_mc_3_turn.json";

    python ./src/sofa/eval/eval_truthfulqa_mc.py --dataset-format "arrow" \
                                        --test-data-file "./data/truthful_qa_mc/data-00000-of-00001.arrow" \
                                        --template "llama-2" \
                                        --model-name "${MODEL_NAME}" \
                                        --model-path "${MODEL_PATH}" \
                                        --system-message "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." \
                                        --k-th-turn 5 \
                                        --output-file-name "truthfulqa_mc_5_turn.json";

done

# test_model=("llama-2-7b_sharegpt_uncensored_wo_sys_align" "llama-2-7b_cd_v5_1106_uncensored_wo_sys_align_reference")

test_model=("llama-2-7b_cd_v5_1106_uncensored_wo_sys_align")

for MODEL_NAME in "${test_model[@]}"
do

    echo ${MODEL_NAME}
    MODEL_PATH="./saved_models/${MODEL_NAME}"

    python ./src/sofa/eval/eval_truthfulqa_mc.py --dataset-format "arrow" \
                                        --test-data-file "./data/truthful_qa_mc/data-00000-of-00001.arrow" \
                                        --template "llama-2" \
                                        --model-name ${MODEL_NAME} \
                                        --model-path ${MODEL_PATH} \
                                        --system-message "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." \
                                        --k-th-turn 3 \
                                        --output-file-name "truthfulqa_mc_3_turn.json";

    python ./src/sofa/eval/eval_truthfulqa_mc.py --dataset-format "arrow" \
                                        --test-data-file "./data/truthful_qa_mc/data-00000-of-00001.arrow" \
                                        --template "llama-2" \
                                        --model-name ${MODEL_NAME} \
                                        --model-path ${MODEL_PATH} \
                                        --system-message "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." \
                                        --k-th-turn 5 \
                                        --output-file-name "truthfulqa_mc_5_turn.json";

done