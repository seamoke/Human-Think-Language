dataset=("numglue" "deepmind" "svamp" "gsm8k" "simuleq")
cuda_id=(3 3 3 3 3)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

for model_path in "gpt-3.5-turbo-16k-0613" "gpt-4-turbo-preview"
do
  for i in "${!dataset[@]}"
  do
    echo ${model_path}
    python to_gpt.py \
    --model ${model_path} \
    --shots 4 \
    --dataset ${dataset[$i]} &
  done
  wait
done
