dataset=('gsm8k')
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

for str in "your_model" 
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