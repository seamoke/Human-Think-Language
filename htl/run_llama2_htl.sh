export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
ENV="codellama"
export NUM_GPUS=$1
export WORKER_NUM=$2
export learning_rate="$3"
elif [ $ENV = codellama ]; then
export MODEL_PATH="MAmmoTH-Coder-7B"
export OUTPUT_PATH="./checkpoint/codellama_final32k${learning_rate}"
export Model_layer="LlamaDecoderLayer"
elif [ $ENV = mistral ]; then
export MODEL_PATH="MAmmoTH-Mistral-7B"
export OUTPUT_PATH="./checkpoint/mistral_final32k${learning_rate}"
export Model_layer="MistralDecoderLayer"
fi


if [[ "$MODEL_PATH" == *7B* ]] || [[ "$MODEL_PATH" == *7b* ]]; then
    per_device_train_batch_size=1
    gradient_accumulation_steps=8
else
    per_device_train_batch_size=1
    gradient_accumulation_steps=8
# If none of the above conditions are true
fi

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export MASTER_ADDR=localhost
export WANDB_PROJECT="Math_model"
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export OMP_NUM_THREADS=12

NODE_RANK=0
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3930
MASTER_PORT="6066"
flash_attn=False

echo -e "MASTER_HOST:$MASTER_HOST\nMASTER_ADDR:$MASTER_ADDR\nNODE_RANK:$NODE_RANK\nWORKER_NUM:$WORKER_NUM\n"
echo -e "NUM_GPUS:$NUM_GPUS\nMODEL_PATH:$MODEL_PATH\nper_device_train_batch_size=$per_device_train_batch_size\ngradient_accumulation_steps=$gradient_accumulation_steps" 
torchrun --master_addr ${MASTER_ADDR} \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=${MASTER_PORT} \
  --nnodes=${WORKER_NUM} \
  --master_addr=${MASTER_ADDR} \
  --node_rank=${NODE_RANK} \
  train_htl.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path "./math_eval/dataset/train_llama_data/llama_data_final32k.jsonl" \
    --bf16 True \
    --output_dir ${OUTPUT_PATH} \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "steps" \
    --eval_steps 0.17 \
    --save_strategy "no" \
    --save_total_limit 3 \
    --learning_rate ${learning_rate} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --save_safetensors True \
    --flash_attn ${flash_attn} \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap ${Model_layer}
