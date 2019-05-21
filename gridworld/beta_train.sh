start_time=$(date +%s)
for ((i=1; i<=3; i++))
do
    for tf in 16 8 4 #len in 32 16 8 4 2 #dim in 8 16 32
    do
        CUDA_VISIBLE_DEVICES=9 python main.py \
        --gpu_fraction 0.1 \
        --train_frequency $tf \
        --image_dim 64 \
        --state_dim 3 \
        --image_padding 1 \
        --weight_q_a 0.5 \
        --autofilter 1 \
        --autolen 1 \
        --hist_len 2 \
        --step_reward -1 \
        --collision_reward -10 \
        --terminal_reward 50 \
        --predict_net "beta" \
        --train_mode "single_beta" \
        --result_dir "ft1_tf"$tf"_"$i
    done
done
end_time=$(date +%s)
echo -e "\n\nTotal time cost: $(($end_time - $start_time))s \n\n"
