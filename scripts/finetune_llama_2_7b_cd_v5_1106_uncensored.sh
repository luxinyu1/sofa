export OMP_NUM_THREADS=1

torchrun --nproc_per_node=8 --master_port=20001 ./src/sofa/train_dialog.py \
    --model_name_or_path ./pretrained_models/hf_llama-2/7B/  \
    --data_path "./data/cd_v5_uncensored_gpt-3.5-turbo-1106.json" \
    --output_dir ./saved_models/llama-2-7b_cd_v5_1106_uncensored/ \
    --template "llama-2" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --metric_for_best_model "loss" \
    --load_best_model_at_end False \
    --report_to "tensorboard" \
    --logging_dir ./logs/llama-2-7b_cd_v5_1106_uncensored/ \
    --seed 42 \
    --bf16 True \
    --tf32 True \
    --log_level "info" \
    --ddp_timeout 7200 \
    --use_flash_attn True \