
### ['gsm8k', 'svamp', 'math', 'numglue', 'deepmind', 'simuleq'] 
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

dataset='mathinstruct'

python ./math_eval/run_data_pot.py \
  --model "MAmmoTH-mistral-7B" \
  --shots 0 \
  --stem_flan_type "" \
  --batch_size 4 \
  --dataset $dataset \
  --model_max_length 1500 \
  --print --use_vllm
