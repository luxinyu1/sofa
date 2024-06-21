K_TH_TURN=1

test_model=("Llama-2-7b-chat-hf" "Llama-2-13b-chat-hf")

for MODEL_NAME in "${test_model[@]}"
do

    echo ${MODEL_NAME}

    MODEL_PATH="./pretrained_models/${MODEL_NAME}"

    python ./src/sofa/eval/eval_constitution_dialog.py --model-name ${MODEL_NAME} \
                                                --model-path ${MODEL_PATH} \
                                                --template "llama-2" \
                                                --k-th-turn ${K_TH_TURN} \
                                                --num-gpus 4 \
                                                --output-file-prefix "hh_rlhf_red_teaming_adv" \
                                                --jailbreak-mode \
                                                --test-data-file "./data/red-team-attempts_1000/data-00000-of-00001.arrow" \
                                                --dataset-format "arrow" \

    python ./src/sofa/eval/eval_constitution_dialog.py --model-name ${MODEL_NAME} \
                                                --model-path ${MODEL_PATH} \
                                                --template "llama-2" \
                                                --k-th-turn ${K_TH_TURN} \
                                                --output-file-prefix "hh_rlhf_red_teaming_adv_wo_sys" \
                                                --remove-sys-prompt \
                                                --jailbreak-mode \
                                                --num-gpus 4 \
                                                --test-data-file "./data/red-team-attempts_1000/data-00000-of-00001.arrow" \
                                                --dataset-format "arrow" \

done

test_model=(
    "llama-2-7b_sharegpt_wo_sys_align" "llama-2-7b_sharegpt_uncensored_wo_sys_align" "llama-2-7b_cd_v5_1106_wo_sys_align" "llama-2-7b_cd_v5_1106_uncensored_wo_sys_align"
    "llama-2-13b_sharegpt_wo_sys_align" "llama-2-13b_sharegpt_uncensored_wo_sys_align" "llama-2-13b_cd_v5_1106_wo_sys_align" "llama-2-13b_cd_v5_1106_uncensored_wo_sys_align"
)

for MODEL_NAME in "${test_model[@]}"
do

    echo $MODEL_NAME

    MODEL_PATH="./saved_models/${MODEL_NAME}"

    python ./src/sofa/eval/eval_constitution_dialog.py --model-name ${MODEL_NAME} \
                                            --model-path ${MODEL_PATH} \
                                            --template "llama-2" \
                                            --k-th-turn ${K_TH_TURN} \
                                            --output-file-prefix "hh_rlhf_red_teaming_adv_wo_sys" \
                                            --remove-sys-prompt \
                                            --jailbreak-mode \
                                            --num-gpus 8 \
                                            --test-data-file "./data/red-team-attempts_1000/data-00000-of-00001.arrow" \
                                            --dataset-format "arrow" ;

    python ./src/sofa/eval/eval_constitution_dialog.py --model-name ${MODEL_NAME} \
                                            --model-path ${MODEL_PATH} \
                                            --template "llama-2" \
                                            --k-th-turn ${K_TH_TURN} \
                                            --output-file-prefix "hh_rlhf_red_teaming_adv" \
                                            --jailbreak-mode \
                                            --num-gpus 8 \
                                            --test-data-file "./data/red-team-attempts_1000/data-00000-of-00001.arrow" \
                                            --dataset-format "arrow" ;
    
done