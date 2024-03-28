#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7
export HF_ENDPOINT=https://hf-mirror.com
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_GID_INDEX=3
#export TORCH_DISTRIBUTED_DETAIL=DEBUG
WORKER_GPU=8
WORKER_0_HOST=localhost
ROLE_INDEX=0
WORKER_0_PORT=12355
WORKER_NUM=1

for insert_args in 0;
do
    for epoch in 1;
    do
        str="your_path/codellama_7B_baseline/"
        name="mistral_cot_pot${epoch}_insert_args_${insert_args}"
        OMP_NUM_THREADS=12 torchrun --nproc_per_node $WORKER_GPU \
        --master_addr $WORKER_0_HOST \
        --node_rank $ROLE_INDEX \
        --master_port $WORKER_0_PORT \
        --nnodes $WORKER_NUM \
        train_llama.py \
            --model_name_or_path "/media/4/longli_models/MAmmoth-coder-7b" \
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
            --learning_rate 1e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --insert_args ${insert_args} \
            --deepspeed "ds_config/ds_config_zero3.json" \
            
    done
done
