export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_GID_INDEX=3
#export TORCH_DISTRIBUTED_DETAIL=DEBUG
WORKER_GPU=7
WORKER_0_HOST=localhost
ROLE_INDEX=0
WORKER_0_PORT=12355
WORKER_NUM=1

for insert_args in 0;
do
    for epoch in 1;
    do
        str="your_path/Mistral_instruct_baseline/"
        name="mistral_cot_pot${epoch}_insert_args_${insert_args}"
        OMP_NUM_THREADS=12 torchrun --nproc_per_node $WORKER_GPU \
        --master_addr $WORKER_0_HOST \
        --node_rank $ROLE_INDEX \
        --master_port $WORKER_0_PORT \
        --nnodes $WORKER_NUM \
        train_mistral.py \
            --model_name_or_path "your_path/Mistral-7B" \
            --data_path "TIGER-Lab/MathInstruct" \
            --fp16 True \
            --output_dir ${str} \
            --num_train_epochs ${epoch} \
            --gradient_checkpointing True \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --run_name ${name} \
            --learning_rate 5e-7 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --insert_args ${insert_args} \
            --deepspeed "ds_config/ds_config_zero3.json" \
            
            # --fsdp "full_shard auto_wrap" \
            # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
            
        #    --tf32 True
    done
done

# ,
#         "zero_quantized_weights": true,
#         "zero_hpz_partition_size": 16,
#         "zero_quantized_gradients": true