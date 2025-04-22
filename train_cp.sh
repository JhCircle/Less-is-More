#!/bin/bash

# Choose strategy: with_few_shot | without_few_shot | average | all
strategy="average"  # ← You can change this to your desired filtering strategy

nohup bash -c "CUDA_VISIBLE_DEVICES=0,1
swift sft \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --train_type lora \
    --output_dir ./CoT_Parsing\
    --dataset 'data/train/${strategy}/training_cot_parsing_role.jsonl' \
    --val_dataset 'data/test/test_cot_parsing_role.jsonl' \
    --num_train_epochs 5 \
    --max_length 3096 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --gradient_checkpointing true \
    --per_device_train_batch_size 4 \
    --weight_decay 0.01 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.03 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --save_only_model true \
    --lorap_lr_ratio 16" > train.log 2>&1 &

tail -f train.log
