model_name=GPT4TS
export CUDA_VISIBLE_DEVICES=1
for percent_aug in 25 50 100; do # 
  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Monthly' \
    --model_id m4_Monthly \
    --model $model_name \
    --data m4 \
    --features M \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --gpt_layer 6 \
    --d_ff 128 \
    --d_model 128 \
    --patch_size 1 \
    --stride 1 \
    --batch_size 16 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.002 \
    --loss 'SMAPE' \
    --aug m4-Monthly-train_168_gen_50ksteps_10k.npy \
    --percent_aug $percent_aug

  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Yearly' \
    --model_id m4_Yearly \
    --model $model_name \
    --data m4 \
    --features M \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --ln 1 \
    --gpt_layer 6 \
    --d_model 768 \
    --d_ff 32 \
    --patch_size 1 \
    --stride 1 \
    --batch_size 16 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --loss 'SMAPE' \
    --aug m4-Yearly-train_168_gen_50ksteps_10k.npy \
    --percent_aug $percent_aug

  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Quarterly' \
    --model_id m4_Quarterly \
    --model $model_name \
    --data m4 \
    --features M \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --gpt_layer 6 \
    --d_model 768 \
    --patch_size 1 \
    --stride 1 \
    --d_ff 128 \
    --batch_size 16 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --loss 'SMAPE' \
    --aug m4-Quarterly-train_168_gen_50ksteps_10k.npy \
    --percent_aug $percent_aug

  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Daily' \
    --model_id m4_Daily \
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
    --aug m4-Daily-train_168_gen_50ksteps_10k.npy \
    --percent_aug $percent_aug


  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
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
    --aug m4-Weekly-train_168_gen_50ksteps_10k.npy \
    --percent_aug $percent_aug

  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Hourly' \
    --model_id m4_Hourly \
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
    --aug m4-Hourly-train_168_gen_50ksteps_10k.npy \
    --percent_aug $percent_aug
done