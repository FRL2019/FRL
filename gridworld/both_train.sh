start_time=$(date +%s)
for ((i=1; i<=6; i++))
do
    for an in False Ture #prob in 0.1 0.2 0.0 #len in 2 4 8 16 32 #dim in 8 16 32
    do
        CUDA_VISIBLE_DEVICES=9 python main.py \
        --train_frequency 1 \
        --image_dim 8 \
        --lambda_ 0.5 \
        --autofilter 1 \
        --autolen 1 \
        --hist_len 2 \
        --add_train_noise $an \
        --add_predict_noise False \
        --noise_prob 0.1 \
        --stddev 1.0 \
        --predict_net "both" \
        --train_mode "frl_separate" \
        --result_dir "an"$an"_prob0.1_std1.0_"$i
    done
done
end_time=$(date +%s)
echo -e "\n\nTotal time cost: $(($end_time - $start_time))s \n\n"
