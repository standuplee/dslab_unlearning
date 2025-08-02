#!/bin/bash

# # Define the parameters
# dataset=("ml-100k")
# model=("wmf")
# group=(10)
# deltype=("interaction")
# delper_values=(0.5 1.0 2.0 5.0 10.0)
# beta=(0.01 0.05 0.1 0.5 1.0 1.5 2.0 2.5 3.0 5.0 10.0)
# verbose=2

# # Original model training
# python main.py --dataset ml-100k --learn retrain --deltype interaction --delper 0.0 --verbose 2

# # Ideal model training

# for delper in ${delper_values[@]}; do
#     python main.py --dataset ml-100k --learn retrain --deltype interaction --delper $delper --verbose 2
# done

# for delper in ${delper_values[@]}; do
#     for beta in ${beta[@]}; do
#             python main_task2.py \
#             --dataset ml-100k \
#             --delper $delper \
#             --beta $beta \
#             --verbose 2 \
#             --origin_model_path model_params/retrain/wmf_ml-100k_interaction_0.0/model.pth \
#             --oracle_model_path model_params/retrain/wmf_ml-100k_interaction_$delper/model.pth \
#             --rank_ratio 1.0
#     done
# done


# Define the parameters
dataset=("ml-100k")
model=("wmf")
group=(10)
deltype=("interaction")
delper_values=(0.5 1.0 2.0 5.0 10.0)
beta=(0.01 0.05 0.1 0.5 1.0 1.5 2.0 2.5 3.0 5.0 10.0)
verbose=2

# Original model training
python main.py --dataset ml-100k --learn retrain --deltype interaction --delper 0.0 --verbose 2

# Ideal model training

for delper in ${delper_values[@]}; do
    python main.py --dataset ml-100k --learn retrain --deltype interaction --delper $delper --verbose 2
done

for delper in ${delper_values[@]}; do
    for beta in ${beta[@]}; do
            python main_task2.py \
            --dataset ml-100k \
            --delper $delper \
            --beta $beta \
            --verbose 2 \
            --origin_model_path model_params/retrain/wmf_gowalla_interaction_0.0/model.pth \
            --oracle_model_path model_params/retrain/wmf_gowalla_interaction_$delper/model.pth \
            --rank_ratio 1.0
    done
done




# Define the parameters
dataset=("adm")
model=("wmf")
group=(10)
deltype=("interaction")
delper_values=(0.5 1.0 2.0 5.0 10.0)
beta=(0.01 0.05 0.1 0.5 1.0 1.5 2.0 2.5 3.0 5.0 10.0)
verbose=2

# Original model training
python main.py --dataset adm --learn retrain --deltype interaction --delper 0.0 --verbose 2

# Ideal model training

for delper in ${delper_values[@]}; do
    python main.py --dataset adm --learn retrain --deltype interaction --delper $delper --verbose 2
done

for delper in ${delper_values[@]}; do
    for beta in ${beta[@]}; do
            python main_task2.py \
            --dataset adm \
            --delper $delper \
            --beta $beta \
            --verbose 2 \
            --origin_model_path model_params/retrain/wmf_adm_interaction_0.0/model.pth \
            --oracle_model_path model_params/retrain/wmf_adm_interaction_$delper/model.pth \
            --rank_ratio 1.0
    done
done




# Define the parameters
dataset=("ml-1m")
model=("wmf")
group=(10)
deltype=("interaction")
delper_values=(0.5 1.0 2.0 5.0 10.0)
beta=(0.01 0.05 0.1 0.5 1.0 1.5 2.0 2.5 3.0 5.0 10.0)
verbose=2

# Original model training
python main.py --dataset ml-1m --learn retrain --deltype interaction --delper 0.0 --verbose 2

# Ideal model training

for delper in ${delper_values[@]}; do
    python main.py --dataset ml-1m --learn retrain --deltype interaction --delper $delper --verbose 2
done

for delper in ${delper_values[@]}; do
    for beta in ${beta[@]}; do
            python main_task2.py \
            --dataset ml-1m \
            --delper $delper \
            --beta $beta \
            --verbose 2 \
            --origin_model_path model_params/retrain/wmf_ml-1m_interaction_0.0/model.pth \
            --oracle_model_path model_params/retrain/wmf_ml-1m_interaction_$delper/model.pth \
            --rank_ratio 1.0
    done
done

