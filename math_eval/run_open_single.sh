dataset=("gsm8k" "svamp" "numglue" "simuleq")
#dataset=("simuleq")
cuda_id=(0 2 1 3 5)
export CUDA_DEVICE_ORDER=PCI_BUS_ID


for i in "${!dataset[@]}"
do
  echo ${dataset[$i]}
  echo ${cuda_id[$i]}
  CUDA_VISIBLE_DEVICES=${cuda_id[$i]}  python run_test_2_stage.py \
  --model "your_pathMAMmoth_checkpoints/MAmmoth_all_format_epoch_1_20000insert" \
  --shots 0 \
  --stem_flan_type "" \
  --batch_size 8 \
  --dataset ${dataset[$i]} \
  --model_max_length 1500 \
  --print \
  --cot_backup &
done

wait