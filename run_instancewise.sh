#!/bin/bash

# 로그 폴더 생성
mkdir -p ./log

# 파라미터 설정
datasets=("ml-100k" "adm")
basemodels=("wmf" "bpr" "lightgcn")
learn_types=("retrain" "sisa" "receraser" "ultrare")
groups=10
deltype="interaction"
delper=5
verbose=2
repeat=5

# 실행 시작
echo "==============================================="
echo "Instance-wise (Edge deletion) 5-run experiments"
echo "Datasets: ${datasets[@]}"
echo "Base Models: ${basemodels[@]}"
echo "Learn Types: ${learn_types[@]}"
echo "==============================================="

for dataset in "${datasets[@]}"; do
  for base_model in "${basemodels[@]}"; do
    for learn in "${learn_types[@]}"; do
      for ((i=1; i<=repeat; i++)); do
        log="./log/${i}_${delper}_${dataset}_${base_model}_${learn}_${deltype}_${groups}.log"

        echo "-----------------------------------------------"
        echo "Run $i / $repeat"
        echo "Dataset: $dataset | Base: $base_model | Learn: $learn"
        echo "-----------------------------------------------"

        python main.py \
          --dataset $dataset \
          --model $base_model \
          --group $groups \
          --learn $learn \
          --deltype $deltype \
          --delper $delper \
          --verbose $verbose | tee $log
      done
    done
  done
done

echo "==============================================="
echo "All runs completed. Logs saved in ./log/"
echo "==============================================="