# python ./src/sofa/eval/k_th_turn_inference_dp.py --model-name "llama-2-7b_sharegpt_wo_sys_align" \
#                                         --model-path "./saved_models/llama-2-7b_sharegpt_wo_sys_align" \
#                                         --probe-file "./data/reference_cd_v5_wo_sys_align_gpt-3.5-turbo-1106.json" \
#                                         --template "llama-2" \
#                                         --k-th-turn 1 \
#                                         --test-aspect "all" \
#                                         --num-gpus 8 \
#                                         --backend "vllm" ;

# python ./src/sofa/eval/k_th_turn_inference_dp.py --model-name "llama-2-13b_sharegpt_uncensored_wo_sys_align" \
#                                         --model-path "./saved_models/llama-2-13b_sharegpt_uncensored_wo_sys_align/" \
#                                         --probe-file "./data/reference_cd_v5_wo_sys_align_gpt-3.5-turbo-1106.json" \
#                                         --template "llama-2" \
#                                         --k-th-turn 1 \
#                                         --test-aspect "all" \
#                                         --num-gpus 8 \
#                                         --backend "vllm" ;

python ./src/sofa/eval/k_th_turn_inference_dp.py --model-name "llama-2-13b_sharegpt_uncensored_wo_sys_align" \
                                        --model-path "./saved_models/llama-2-13b_sharegpt_uncensored_wo_sys_align/" \
                                        --probe-file "./data/reference_cd_v5_wo_sys_align_gpt-3.5-turbo-1106.json" \
                                        --template "llama-2" \
                                        --k-th-turn 1 \
                                        --test-aspect "all" \
                                        --num-gpus 8 \
                                        --backend "vllm" ;

python ./src/sofa/eval/k_th_turn_inference_dp.py --model-name "llama-2-7b_sharegpt_uncensored_wo_sys_align" \
                                        --model-path "./saved_models/llama-2-7b_sharegpt_uncensored_wo_sys_align/" \
                                        --probe-file "./data/reference_cd_v5_wo_sys_align_gpt-3.5-turbo-1106.json" \
                                        --template "llama-2" \
                                        --k-th-turn 1 \
                                        --test-aspect "all" \
                                        --num-gpus 8 \
                                        --backend "vllm" ;