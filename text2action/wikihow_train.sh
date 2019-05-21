start_time=$(date +%s)
for ((i=1; i<=1; i++))
do
    for pn in True #std in 0.1 0.5 1.0 #prob in 0.25 0.5 1.0 #fw in 32 
    do
        CUDA_VISIBLE_DEVICES=0 python main.py \
        --domain "wikihow" \
        --agent_mode "arg" \
        --gpu_fraction 0.2 \
        --model_dim 50 \
        --lambda_ 0.5 \
        --num_filters 32 \
        --k_fold 5 \
        --preset_lambda False \
        --add_train_noise True \
        --add_predict_noise $pn \
        --noise_prob 0.5 \
        --stddev 1.0 \
        --predict_net "both" \
        --train_mode "frl_separate" \
        --result_dir "_anTrue_pn"$pn"_"$i 
    done
done
end_time=$(date +%s)
echo -e "\n\nTotal time cost: $(($end_time - $start_time))s \n\n"
