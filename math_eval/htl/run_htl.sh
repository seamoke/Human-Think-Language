dataset=('gsm8k' 'svamp' 'math')
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

for str in  "htl_model"
do
  model_path=${str}
  #model_path="${model_base}${str}"
  echo $model_path

  for i in ${dataset[@]}
  do
    python run_open_htl.py \
    --model $model_path \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset ${i} \
    --model_max_length 3072 \
    --cot_backup \
    --print  --use_vllm
  done
done