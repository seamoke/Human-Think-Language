
### ['gsm8k', 'svamp', 'math', 'numglue', 'deepmind', 'simuleq'] 
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

dataset='mathinstruct'

python ./math_eval/run_data_cot.py \
  --model "MAmmoTH-mistral-7B" \
  --shots 0 \
  --stem_flan_type "pot_prompt" \
  --batch_size 4 \
  --dataset $dataset \
  --model_max_length 2048 \
  --print \
  --use_vllm
