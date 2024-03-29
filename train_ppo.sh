export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
#export CUDA_VISIBLE_DEVICES=1,2
#export CUDA_VISIBLE_DEVICES=5
# OMP_NUM_THREADS=12 accelerate launch --config_file "/media/2/seamoke/new_MAmmoth/llama-trl/configs/deepspeed.yaml" --num_machines 1  --num_processes 8 --num_cpu_threads_per_process 24 --mixed_precision fp16\
#  tuning_lm_with_rl.py \
#     --log_with wandb \
#     --model_name /media/4/longli_models/MAMmoth_checkpoints/new_insert/MAmmoth_all_data/1/all_insert_args_0.3 \
#     --reward_model_name /media/4/longli_models/LORA_LLAMA/training_reward_model/peft_last_checkpoint \
#     --adafactor False \
#     --tokenizer_name /media/4/longli_models/MAMmoth_checkpoints/new_insert/MAmmoth_all_data/1/all_insert_args_0.3 \
#     --save_freq 400 \
#     --output_max_length 512 \
#     --batch_size 8 \
#     --dataset_name ./data/llama_tune.json \
#     --gradient_accumulation_steps 4 \
#     --batched_gen True \
#     --ppo_epochs 4 \
#     --learning_rate 1e-5 \
#     --early_stopping True \
#     --output_dir './checkpoints/tuning_llama_rl/'
OMP_NUM_THREADS=12 accelerate launch --multi_gpu --num_machines 1  --num_processes 2 --num_cpu_threads_per_process 24 --mixed_precision fp16\
 tuning_lm_with_rl.py \
    --log_with wandb \
    --model_name /media/4/longli_models/MAMmoth_checkpoints/new_insert/MAmmoth_all_data/1/all_insert_args_0.3 \
    --reward_model_name /media/4/longli_models/LORA_LLAMA/training_reward_model/peft_last_checkpoint \
    --adafactor False \
    --tokenizer_name /media/4/longli_models/MAMmoth_checkpoints/new_insert/MAmmoth_all_data/1/all_insert_args_0.3 \
    --save_freq 400 \
    --output_max_length 512 \
    --batch_size 8 \
    --dataset_name ./data/llama_tune.json \
    --gradient_accumulation_steps 4 \
    --batched_gen True \
    --ppo_epochs 4 \
    --learning_rate 1e-5 \
    --early_stopping True \
    --output_dir './checkpoints/tuning_llama_rl/'

# python tuning_lm_with_rl.py \
#     --log_with wandb \
#     --model_name /media/4/longli_models/MAMmoth_checkpoints/new_insert/MAmmoth_all_data/1/all_insert_args_0.3 \
#     --reward_model_name /media/4/longli_models/LORA_LLAMA/training_reward_model/peft_last_checkpoint \
#     --adafactor False \
#     --tokenizer_name /media/4/longli_models/MAMmoth_checkpoints/new_insert/MAmmoth_all_data/1/all_insert_args_0.3 \
#     --save_freq 400 \
#     --output_max_length 1024 \
#     --batch_size 8 \
#     --dataset_name ./data/llama_tune.json \
#     --gradient_accumulation_steps 4 \
#     --batched_gen True \
#     --ppo_epochs 4 \
#     --learning_rate 1e-5 \
#     --early_stopping True \
#     --output_dir './checkpoints/tuning_llama_rl/'