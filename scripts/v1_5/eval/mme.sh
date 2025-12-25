#!/bin/bash

# 设置单卡运行
CUDA_VISIBLE_DEVICES=0

# 模型与数据配置
CKPT="llava-v1.5-7b"
CKPT_DIR="./checkpoints/${CKPT}"
DATA_DIR="./playground/data/eval"
SPLIT="llava_mme"

# 获取 token 数，默认128
TOKEN=${1:-128}
PARAM="118_vtn_${TOKEN}"

# 输出路径
ANS_DIR="${DATA_DIR}/MME/answers/${SPLIT}/${CKPT}"
ANS_FILE="${ANS_DIR}/${PARAM}.jsonl"
mkdir -p "${ANS_DIR}"

# 推理阶段
python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR} \
    --question-file ${DATA_DIR}/MME/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/MME/MME_Benchmark_release_version \
    --answers-file ${ANS_FILE} \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

# 转换答案为 MME 评估格式
cd ${DATA_DIR}/MME

python convert_answer_to_mme.py \
    --experiment ${SPLIT}/${CKPT}/${PARAM}

# 执行评估脚本
cd eval_tool

python calculation.py --results_dir answers/${SPLIT}/${CKPT}/${PARAM}
# python convert_answer_to_mme.py --experiment "llava_mme/llava-v1.5-7b/vtn_64"
# python calculation.py --results_dir "answers/llava_mme/llava-v1.5-7b/118_vtn_128"