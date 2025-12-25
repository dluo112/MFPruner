#!/bin/bash

# 设置单卡运行
CUDA_VISIBLE_DEVICES=0

# 模型与数据路径（请根据实际路径修改）
CKPT="llava-v1.5-7b"
CKPT_DIR="./checkpoints/${CKPT}"
DATA_DIR="./playground/data/eval"
SPLIT="llava_textvqa_val_v051_ocr"

# 可视 token 数（命令行传入或默认128）
TOKEN=${1:-128}
PARAM="vtn_${TOKEN}"

# 输出路径
ANS_DIR="${DATA_DIR}/textvqa/answers/${SPLIT}/${CKPT}"
mkdir -p "${ANS_DIR}"
ANS_FILE="${ANS_DIR}/${PARAM}.jsonl"

# 推理阶段
python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR} \
    --question-file ${DATA_DIR}/textvqa/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/textvqa/train_images \
    --answers-file ${ANS_FILE} \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

# 评估阶段
python -m llava.eval.eval_textvqa \
    --annotation-file ${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${ANS_FILE}
