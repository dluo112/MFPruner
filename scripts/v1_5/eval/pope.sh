#!/bin/bash

CKPT_DIR="./checkpoints"
DATA_DIR="./playground/data/eval"

CKPT="llava-v1.5-7b"
SPLIT="llava_pope_test"

TOKEN=${1}
PARAM="vote3_vtn_${TOKEN}"

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ${DATA_DIR}/pope/coco \
    --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
    --result-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl
# python llava/eval/eval_pope.py \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file ./playground/data/eval/pope/answers/llava_pope_test/llava-v1.5-7b/118_vtn_32.jsonl
