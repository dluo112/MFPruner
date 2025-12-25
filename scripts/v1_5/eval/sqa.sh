#!/bin/bash

# 指定 GPU（单卡）
CUDA_VISIBLE_DEVICES=0

# 模型和数据集配置（请根据需要修改路径）
CKPT="llava-v1.5-7b"
CKPT_DIR="./checkpoints/${CKPT}"
DATA_DIR="./playground/data/eval"
SPLIT="llava_test_CQM-A"

# 可视 token 数（支持命令行传参，默认128）
TOKEN=${1:-128}
PARAM="SIM_vtn_${TOKEN}"

# 输出路径
ANS_DIR="${DATA_DIR}/scienceqa/answers/${SPLIT}/${CKPT}"
mkdir -p "${ANS_DIR}"

ANS_FILE="${ANS_DIR}/${PARAM}.jsonl"
OUTPUT_FILE="${ANS_DIR}/${PARAM}_output.jsonl"
RESULT_FILE="${ANS_DIR}/${PARAM}_result.json"

# 推理阶段
python -m llava.eval.model_vqa_science \
    --model-path ${CKPT_DIR} \
    --question-file ${DATA_DIR}/scienceqa/${SPLIT}.json \
    --image-folder ${DATA_DIR}/scienceqa/test \
    --answers-file ${ANS_FILE} \
    --visual_token_num ${TOKEN} \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

# 评估阶段
python -m llava.eval.eval_science_qa \
    --base-dir ${DATA_DIR}/scienceqa \
    --result-file ${ANS_FILE} \
    --output-file ${OUTPUT_FILE} \
    --output-result ${RESULT_FILE}
