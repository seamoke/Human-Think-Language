
### ['gsm8k', 'svamp', 'math', 'numglue', 'deepmind', 'simuleq'] 
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

dataset='mathinstruct'

python run_data_insert.py \
  --model "/media/4/longli_models/Mammoth-mistral-7b" \
  --shots 0 \
  --stem_flan_type "" \
  --batch_size 4 \
  --dataset $dataset \
  --model_max_length 1500 \
  --print \
  --cot_backup \
