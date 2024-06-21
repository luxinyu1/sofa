API_KEY=""
API_BASE=""
RES_FILE=""

python ./src/sofa/eval/eval_openai_red_teaming.py --model "gpt-3.5-turbo" \
                                                --api_key ${API_KEY} \
                                                --api_base ${API_BASE} \
                                                --assistant_res_file ${RES_FILE}