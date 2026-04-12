#!/bin/bash
set -e

# 配置
MODEL="or/gpt-4o-mini"
PY="python3"

echo "=========================================="
echo " E8 最简运行脚本（无日志、跳过baseline）"
echo " 模型：$MODEL"
echo "=========================================="

echo -e "\n➡️  运行 E8：重排序"
$PY -m experiments.e8_retrieval_enhancement --mode rerank --model $MODEL

echo -e "\n➡️  运行 E8：全增强（改写+重排序）"
$PY -m experiments.e8_retrieval_enhancement --mode full --model $MODEL

echo -e "\n=========================================="
echo " ✅ ALL DONE —— 全部实验跑完！"
echo "=========================================="