#!/bin/bash

# Define the parameters
dataset=("ml-100k")
model=("wmf")
group=(10)
deltype=("interaction")
delper_values=(0.5 1.0 2.0 5.0 10.0)
alpha=(0.5 1.0 1.5 2.0 2.5 3.0)
beta=(0.5 1.0 1.5 2.0 2.5 3.0)
rank_ratio=(0.5 0.6 0.7 0.8 0.9 1.0)
use_degree_weighting=(True False)
degree_weight_power=(0.5)
normalize_weights=(True False)
verbose=2


# Original model training
python main.py --dataset ml-100k --learn retrain --deltype interaction --delper 0.0 --verbose 2

# Ideal model training

for delper in ${delper_values[@]}; do
    python main.py --dataset ml-100k --learn retrain --deltype interaction --delper $delper --verbose 2
done

# Task vector model training & params tuning

for delper in ${delper_values[@]}; do
    for alpha in ${alpha[@]}; do
        for beta in ${beta[@]}; do
            for rank_ratio in ${rank_ratio[@]}; do
                for use_degree_weighting in ${use_degree_weighting[@]}; do
                    for degree_weight_power in ${degree_weight_power[@]}; do
                        for normalize_weights in ${normalize_weights[@]}; do
                        python main_task.py \
            --dataset ml-100k \
            --delper $delper \
            --alpha $alpha \
            --beta $beta \
            --verbose 2 \
            --origin_model_path model_params/retrain/wmf_ml-100k_interaction_0.0/model.pth \
            --oracle_model_path model_params/retrain/wmf_ml-100k_interaction_$delper/model.pth \
            --rank_ratio $rank_ratio \
            --use_degree_weighting $use_degree_weighting \
            --degree_weight_power $degree_weight_power \
            --normalize_weights $normalize_weights
                        done
                    done
                done
            done
        done
    done
done



##########################################################


# Original model training
python main.py --dataset gowalla --learn retrain --deltype interaction --delper 0 --verbose 2

# Ideal model training

for delper in ${delper_values[@]}; do
    python main.py --dataset gowalla --learn retrain --deltype interaction --delper $delper --verbose 2
done

# Task vector model training & params tuning

for delper in ${delper_values[@]}; do
    for alpha in ${alpha[@]}; do
        for beta in ${beta[@]}; do
            for rank_ratio in ${rank_ratio[@]}; do
                for use_degree_weighting in ${use_degree_weighting[@]}; do
                    for degree_weight_power in ${degree_weight_power[@]}; do
                        for normalize_weights in ${normalize_weights[@]}; do
                        python main_task.py \
            --dataset gowalla \
            --delper $delper \
            --alpha $alpha \
            --beta $beta \
            --verbose 2 \
            --origin_model_path model_params/retrain/wmf_gowalla_interaction_0.0/model.pth \
            --oracle_model_path model_params/retrain/wmf_gowalla_interaction_$delper/model.pth \
            --rank_ratio $rank_ratio \
            --use_degree_weighting $use_degree_weighting \
            --degree_weight_power $degree_weight_power \
            --normalize_weights $normalize_weights
                        done
                    done
                done
            done
        done
    done
done




##########################################################


##########################################################


# Original model training
python main.py --dataset adm --learn retrain --deltype interaction --delper 0 --verbose 2

# Ideal model training

for delper in ${delper_values[@]}; do
    python main.py --dataset adm --learn retrain --deltype interaction --delper $delper --verbose 2
done

# Task vector model training & params tuning

for delper in ${delper_values[@]}; do
    for alpha in ${alpha[@]}; do
        for beta in ${beta[@]}; do
            for rank_ratio in ${rank_ratio[@]}; do
                for use_degree_weighting in ${use_degree_weighting[@]}; do
                    for degree_weight_power in ${degree_weight_power[@]}; do
                        for normalize_weights in ${normalize_weights[@]}; do
                        python main_task.py \
            --dataset adm \
            --delper $delper \
            --alpha $alpha \
            --beta $beta \
            --verbose 2 \
            --origin_model_path model_params/retrain/wmf_adm_interaction_0.0/model.pth \
            --oracle_model_path model_params/retrain/wmf_adm_interaction_$delper/model.pth \
            --rank_ratio $rank_ratio \
            --use_degree_weighting $use_degree_weighting \
            --degree_weight_power $degree_weight_power \
            --normalize_weights $normalize_weights
                        done
                    done
                done
            done
        done
    done
done


##########################################################


# Original model training
python main.py --dataset ml-1m --learn retrain --deltype interaction --delper 0 --verbose 2

# Ideal model training

for delper in ${delper_values[@]}; do
    python main.py --dataset ml-1m --learn retrain --deltype interaction --delper $delper --verbose 2
done

# Task vector model training & params tuning

for delper in ${delper_values[@]}; do
    for alpha in ${alpha[@]}; do
        for beta in ${beta[@]}; do
            for rank_ratio in ${rank_ratio[@]}; do
                for use_degree_weighting in ${use_degree_weighting[@]}; do
                    for degree_weight_power in ${degree_weight_power[@]}; do
                        for normalize_weights in ${normalize_weights[@]}; do
                        python main_task.py \
            --dataset ml-1m \
            --delper $delper \
            --alpha $alpha \
            --beta $beta \
            --verbose 2 \
            --origin_model_path model_params/retrain/wmf_ml-1m_interaction_0.0/model.pth \
            --oracle_model_path model_params/retrain/wmf_ml-1m_interaction_$delper/model.pth \
            --rank_ratio $rank_ratio \
            --use_degree_weighting $use_degree_weighting \
            --degree_weight_power $degree_weight_power \
            --normalize_weights $normalize_weights
                        done
                    done
                done
            done
        done
    done
done
