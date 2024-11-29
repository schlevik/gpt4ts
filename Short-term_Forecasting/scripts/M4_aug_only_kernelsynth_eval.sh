model_name=GPT4TS
for percent_aug in 100; do # 
  python -u run.py \
    --task_name short_term_forecast \
    --is_training 0 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Weekly' \
    --model_id m4_Weekly \
    --model $model_name \
    --data m4 \
    --features M \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --gpt_layer 6 \
    --d_model 768 \
    --d_ff 128 \
    --patch_size 1 \
    --stride 1 \
    --batch_size 16 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --loss 'SMAPE' \
    --aug kernelsynth.npy \
    --percent_aug $percent_aug \
    --aug_only 1
done