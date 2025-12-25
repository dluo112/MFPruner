#!/bin/bash

# 指定使用单卡
CUDA_VISIBLE_DEVICES=0

# 模型和数据配置（根据实际路径修改）
CKPT="llava-v1.5-7b"
CKPT_DIR="./checkpoints/${CKPT}"
DATA_DIR="./playground/data/eval"
SPLIT="mmbench_dev_20230712"

# 获取可视 token 数，默认128
TOKEN=${1:-128}
PARAM="Relevance_vtn_${TOKEN}"

# 输出路径
ANS_DIR="${DATA_DIR}/mmbench/answers/${SPLIT}/${CKPT}"
UPLOAD_DIR="${DATA_DIR}/mmbench/answers_upload/${SPLIT}/${CKPT}"
mkdir -p "${ANS_DIR}"
mkdir -p "${UPLOAD_DIR}"

ANS_FILE="${ANS_DIR}/${PARAM}.jsonl"

# 推理阶段
python -m llava.eval.model_vqa_mmbench \
    --model-path ${CKPT_DIR} \
    --question-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
    --answers-file ${ANS_FILE} \
    --visual_token_num ${TOKEN} \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

# 转换为提交格式
python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
    --result-dir ${ANS_DIR} \
    --upload-dir ${UPLOAD_DIR} \
    --experiment ${PARAM}
# python scripts/convert_mmbench_for_submission.py \
#     --annotation-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
#     --result-dir "./playground/data/eval/mmbench/answers/mmbench_dev_20230712/llava-v1.5-7b" \
#     --upload-dir "./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712/llava-v1.5-7b" \
#     --experiment "vtn_32"


#     CKPT="llava-v1.5-7b"
# CKPT_DIR="./checkpoints/${CKPT}"
# DATA_DIR="./playground/data/eval"
# SPLIT="mmbench_dev_20230712"
# ANS_DIR="./playground/data/eval/mmbench/answers/mmbench_dev_20230712/llava-v1.5-7b"
# UPLOAD_DIR="./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712/llava-v1.5-7b"