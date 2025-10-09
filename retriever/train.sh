nproc_per_node=<NUM_GPUS>
NPROC_PER_NODE=$nproc_per_node \
INFONCE_TEMPERATURE=0.01 \
INFONCE_HARD_NEGATIVES=8 \
swift sft \
    --model /path/to/qwen3-embedding-8b \
    --task_type embedding \
    --model_type qwen3_emb \
    --dataset_num_proc 32 \
    --train_type full \
    --dataset /path/to/training_data.jsonl \
    --split_dataset_ratio 0 \
    --eval_strategy steps \
    --output_dir /path/to/output \
    --eval_steps 1000000 \
    --num_train_epochs 3 \
    --save_steps 800 \
    --per_device_train_batch_size <BATCH_SIZE> \
    --per_device_eval_batch_size 20 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --max_length 4608 \
    --attn_impl flash_attention_2 \
    --use_liger_kernel true
