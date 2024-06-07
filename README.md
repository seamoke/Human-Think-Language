
# **Human Think Language** 
This repo contains the code and data for "[How Do Humans Write Code? Large Models Do It the Same Way Too]"

# How to Use
How to test the probability of CoT (Chain of Thought) errors and PoT (Prompt of Thought) errors on various datasets.
```bash
cd math_eval
```
## 
```bash 
dataset=('gsm8k')
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
for str in "your_model_path" 
do
  model_path=${str}
  echo $model_path
  for i in ${dataset[@]}
  do
    python run_pre.py \
    --model $model_path \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset ${i} \
    --model_max_length 2048 \
    --print --use_vllm
  done
done
```
How to test in GPT-4-turbo
```
bash to_gpt.sh
```
## 0. Get Baseline model
You can download the model from https://huggingface.co/TIGER-Lab/MAmmoTH-Coder-7B and https://huggingface.co/TIGER-Lab/MAmmoTH-7B-Mistral \
Note: MAmmoTH-Coder use transformers==4.31.0, there has some difference.\
if you use 4.31.0, maybe need to install \
We provide two llama_model.py:\
./htl/llama_model.py : transformers==4.31.0 \
./htl/new_llama.py : transformers==4.39.2 
## 1. Get Data
Replace all the "your_path" to the path of your model.
I have construct llama data in math_eval/dataset/llama_data and math_eval/dataset/mistral
You can get more data by 
```bash
cd math_eval
bash run_data_gsm.sh
bash run_data_gsm_pot.sh
```
## 2.Train
```bash
bash htl/run_llama2_htl.sh $NUM_GPUS $gpu_node $lr
example:
bash htl/run_llama2_htl.sh 8 1 2e-5
```
or
```bash
# deepspeed zero3
WORKER_GPU=8
WORKER_0_HOST=localhost
ROLE_INDEX=0
WORKER_0_PORT=12355
WORKER_NUM=1
OMP_NUM_THREADS=12 torchrun --nproc_per_node $WORKER_GPU \
        --master_addr $WORKER_0_HOST \
        --node_rank $ROLE_INDEX \
        --master_port $WORKER_0_PORT \
        --nnodes $WORKER_NUM \
        train_llama.py \
            --model_name_or_path "your_path/MAmmoth-coder-7b" \
            --data_path "dataset_path" \
            --fp16 True \
            --output_dir ${str} \
            --num_train_epochs ${epoch} \
            --gradient_checkpointing True \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --run_name ${name} \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --insert_args ${insert_args} \
            --deepspeed "ds_config/ds_config_zero3.json" \
```
For PPO stage
You should get code and environment from https://github.com/jasonvanf/llama-trl. Then replace tuning_lm_with_rl.py.
```bash
bash train_ppo.sh
```
## 3.Eval
```bash
cd math_eval
```
```bash
dataset="deepmind"
#dataset=("gsm8k" "svamp" "numglue" "simuleq")
model_path="your model"
echo ${model_path}
echo ${dataset[$i]}
python run_test_open.py \
--model ${model_path} \
--shots 0 \
--stem_flan_type "pot_prompt" \
--batch_size 4 \
--dataset ${dataset} \
--model_max_length 1500 \
--print \
--cot_backup 
```
if you want to use vllm
```bash
dataset="deepmind"
#dataset=("gsm8k" "svamp" "numglue" "simuleq")
model_path="your model"
echo ${model_path}
echo ${dataset[$i]}
python run_test_open.py \
--model ${model_path} \
--shots 0 \
--stem_flan_type "pot_prompt" \
--batch_size 4 \
--dataset ${dataset} \
--model_max_length 1500 \
--print \
--cot_backup 
--use_vllm --gpus 8

```
