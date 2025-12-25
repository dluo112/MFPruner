#!/bin/bash

# 模型及数据路径
CKPT="llava-v1.5-7b"
CKPT_DIR="./checkpoints/${CKPT}"
SPLIT="llava_vqav2_mscoco_test-dev2015"
DATA_DIR="./playground/data/eval"

# 可视token个数：命令行传入，否则默认128
TOKEN=${1:-128}
PARAM="vtn_${TOKEN}"

# 输出路径
ANS_DIR="${DATA_DIR}/vqav2/answers/${SPLIT}/${CKPT}/${PARAM}"
mkdir -p "$ANS_DIR"
ANS_FILE="${ANS_DIR}/merge.jsonl"

# 模型推理
python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR} \
    --question-file ${DATA_DIR}/vqav2/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/vqav2/test2015 \
    --answers-file ${ANS_FILE} \
    --num-chunks 1 \
    --chunk-idx 0 \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

# 格式转换为上传文件
python scripts/convert_vqav2_for_submission.py \
    --dir ${DATA_DIR}/vqav2 \
    --src answers/${SPLIT}/${CKPT}/${PARAM}/merge.jsonl \
    --dst answers_upload/${SPLIT}-${CKPT}-${PARAM}.json
