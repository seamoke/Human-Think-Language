
### ['gsm8k', 'svamp', 'math', 'numglue', 'deepmind', 'simuleq'] 
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

dataset='numglue'

proxychains4 python run_data_insert.py \
  --model "your_pathMAmmoth-coder-7b" \
  --shots 0 \
  --stem_flan_type "" \
  --batch_size 8 \
  --dataset $dataset \
  --model_max_length 1500 \
  --print \
  --cot_backup \
