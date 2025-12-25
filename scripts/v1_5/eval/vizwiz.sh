#!/bin/bash

# 设置使用的 GPU（默认第0张）
CUDA_VISIBLE_DEVICES=0

# 设置模型和数据路径（请根据实际情况修改）
CKPT="llava-v1.5-7b"
CKPT_DIR="./checkpoints/${CKPT}"
DATA_DIR="./playground/data/eval"

# 数据划分（可根据需要更换）
SPLIT="llava_test"

# 可视 token 数量（从命令行获取，默认128）
TOKEN=${1:-128}
PARAM="vtn_${TOKEN}"

# 输出路径设置
ANS_DIR="${DATA_DIR}/vizwiz/answers/${SPLIT}/${CKPT}/${PARAM}"
mkdir -p "$ANS_DIR"
ANS_FILE="${ANS_DIR}.jsonl"

# 模型推理阶段
python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR} \
    --question-file ${DATA_DIR}/vizwiz/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/vizwiz/test \
    --answers-file ${ANS_FILE} \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

# 结果转换为上传格式
python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ${DATA_DIR}/vizwiz/${SPLIT}.jsonl \
    --result-file ${ANS_FILE} \
    --result-upload-file ${DATA_DIR}/vizwiz/answers_upload/${SPLIT}/${CKPT}/${PARAM}.json
