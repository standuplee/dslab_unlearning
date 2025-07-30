#!/bin/bash

# # Define the parameters
# dataset=("ml-100k")
# model=("wmf")
# group=(10)
# learn=("retrain")
# deltype=("random")
# delper=(5)
# verbose=2
# # Construct the command
# log="./log/${delper}_${dataset}_${model}_${learn}_${deltype}_${group}.txt"
# cmd="nohup python main.py --dataset $dataset --model $model --group $group --learn $learn --deltype $deltype --delper $delper --verbose $verbose > $log 2>&1 &"

# # run lightGCN
# # cmd="nohup python lightgcn.py --dataset $dataset --group $group --learn $learn --deltype $deltype --delper $delper --verbose $verbose > $log 2>&1 &"

# # Print and execute the command
# echo "Running: sh"
# eval $cmd

# 로그 폴더 생성
mkdir -p ./log

# 파라미터 설정
dataset="ml-100k"
model="wmf"
group=10
learn="retrain"
deltype="random"
delper=5
verbose=2

# 로그 파일 경로
log="./log/${delper}_${dataset}_${model}_${learn}_${deltype}_${group}.log"

# 실행 정보 출력
echo "==============================================="
echo "Running Experiment"
echo "Dataset: $dataset"
echo "Model: $model"
echo "Learn type: $learn"
echo "Del type: $deltype ($delper%)"
echo "Groups: $group"
echo "Verbose: $verbose"
echo "Log file: $log"
echo "==============================================="

# **실시간 콘솔 출력 + 파일 저장**
python main.py \
  --dataset $dataset \
  --model $model \
  --group $group \
  --learn $learn \
  --deltype $deltype \
  --delper $delper \
  --verbose $verbose | tee $log