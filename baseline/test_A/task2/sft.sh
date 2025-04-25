NPROC_PER_NODE=1
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen2_5-7b-instruct \
    --model_id_or_path model/path \
    --model_revision master \
    --sft_type lora \
    --tuner_backend swift \
    --template_type qwen \
    --dtype AUTO \
    --output_dir ./model \
    --custom_train_dataset_path 'train_data/train.jsonl' \
    --num_train_epochs 5 \
    --max_length 3072 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 8 / $NPROC_PER_NODE) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --use_flash_attn false \
    --save_only_model true