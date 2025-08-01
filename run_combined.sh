#!/bin/bash

# 로그 폴더 생성
mkdir -p ./log

# 설정
dataset="ml-100k"
model="wmf"
group=10
learn_type_list=("instance" "task_vector" "joint_task_vector")
learn_methods_instance=("retrain" "sisa" "ultrare" "recformer")
deltype="interaction"
delper_values=(0.5 1.0 2.0 5.0 10.0)
verbose=2

# 실행
for learn_type in "${learn_type_list[@]}"; do
  if [ "$learn_type" == "instance" ]; then
    for learn in "${learn_methods_instance[@]}"; do
      for delper in "${delper_values[@]}"; do
        log="./log/${dataset}_${model}_${learn_type}_${learn}_${deltype}_${delper}.log"
        echo "==============================================="
        echo "Running $learn_type | $learn: delper=$delper"
        echo "Log: $log"
        echo "==============================================="
        
        python main_combined.py \
          --dataset $dataset \
          --model $model \
          --group $group \
          --learn_type $learn_type \
          --learn $learn \
          --deltype $deltype \
          --delper $delper \
          --verbose $verbose | tee $log
      done
    done
  else
    # Task Vector & Joint Task Vector (learn 고정)
    learn="retrain"
    for delper in "${delper_values[@]}"; do
      log="./log/${dataset}_${model}_${learn_type}_${learn}_${deltype}_${delper}.log"
      echo "==============================================="
      echo "Running $learn_type | $learn: delper=$delper"
      echo "Log: $log"
      echo "==============================================="
      
      python main_combined.py \
        --dataset $dataset \
        --model $model \
        --group $group \
        --learn_type $learn_type \
        --learn $learn \
        --deltype $deltype \
        --delper $delper \
        --verbose $verbose | tee $log
    done
  fi
done