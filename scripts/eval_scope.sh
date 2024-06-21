MODEL_NAME=llama-2-7b_cd_v5_1106_uncensored_wo_sys_align
MODEL_PATH=./saved_models/${MODEL_NAME}

python ./src/sofa/eval/eval_scope.py --model-name ${MODEL_NAME} \
                                    --model-path ${MODEL_PATH} \
                                    --template "llama-2" \
                                    --k-th-turn 1 \
                                    --num-gpus 4 \

# python ./src/sofa/eval/eval_scope.py --model-name ${MODEL_NAME} \
#                                     --model-path ${MODEL_PATH} \
#                                     --template "llama-2" \
#                                     --k-th-turn 1 \
#                                     --num-gpus 4 \
#                                     --prepend-system-message-as-user \
#                                     --output-file-name "scope_sys_as_user.json"