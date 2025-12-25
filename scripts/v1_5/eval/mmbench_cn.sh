#!/bin/bash

# 设置使用的 GPU（默认单卡）
CUDA_VISIBLE_DEVICES=0

# 模型与数据路径配置（根据实际修改）
CKPT="llava-v1.5-7b"
CKPT_DIR="./checkpoints/${CKPT}"
DATA_DIR="./playground/data/eval"
SPLIT="mmbench_dev_cn_20231003"

# 可视 token 数量（从命令行读取，默认128）
TOKEN=${1:-128}
PARAM="vtn_${TOKEN}"

# 输出路径
ANS_DIR="${DATA_DIR}/mmbench_cn/answers/${SPLIT}/${CKPT}"
UPLOAD_DIR="${DATA_DIR}/mmbench_cn/answers_upload/${SPLIT}/${CKPT}"
mkdir -p "${ANS_DIR}"
mkdir -p "${UPLOAD_DIR}"

ANS_FILE="${ANS_DIR}/${PARAM}.jsonl"

# 推理阶段
python -m llava.eval.model_vqa_mmbench \
    --model-path ${CKPT_DIR} \
    --question-file ${DATA_DIR}/mmbench_cn/${SPLIT}.tsv \
    --answers-file ${ANS_FILE} \
    --visual_token_num ${TOKEN} \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

# 转换为上传格式
python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_DIR}/mmbench_cn/${SPLIT}.tsv \
    --result-dir ${ANS_DIR} \
    --upload-dir ${UPLOAD_DIR} \
    --experiment ${PARAM}
