#!/bin/bash

# 设置使用的 GPU
export CUDA_VISIBLE_DEVICES=0

# 模型与数据配置
CKPT="llava-v1.5-7b"
CKPT_DIR="./checkpoints/${CKPT}"
DATA_DIR="./playground/data/eval"
SPLIT="llava_gqa_testdev_balanced"

# 视觉 token 数
TOKEN=${1:-128}
PARAM="vtn_${TOKEN}"

# 输出路径配置
ANS_DIR="${DATA_DIR}/gqa/answers/${SPLIT}/${CKPT}/${PARAM}"
mkdir -p "${ANS_DIR}"
ANS_FILE="${ANS_DIR}/merge.jsonl"

# 检查必要文件是否存在
if [ ! -d "${CKPT_DIR}" ]; then
    echo "Error: Model checkpoint directory not found: ${CKPT_DIR}"
    exit 1
fi

if [ ! -f "${DATA_DIR}/gqa/${SPLIT}.jsonl" ]; then
    echo "Error: Question file not found: ${DATA_DIR}/gqa/${SPLIT}.jsonl"
    exit 1
fi

# 推理阶段
echo "Starting inference..."
python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR} \
    --question-file ${DATA_DIR}/gqa/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/gqa/images \
    --answers-file ${ANS_FILE} \
    --num-chunks 1 \
    --chunk-idx 0 \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

# 检查推理是否成功
if [ ! -f "${ANS_FILE}" ]; then
    echo "Error: Inference failed, answers file not generated"
    exit 1
fi

# 转换为 GQA 官方评估格式
echo "Converting to GQA format..."
python scripts/convert_gqa_for_eval.py \
    --src ${ANS_FILE} \
    --dst ${DATA_DIR}/gqa/testdev_balanced_predictions.json

# 评估
echo "Starting evaluation..."
cd ${DATA_DIR}/gqa

# 确保评估脚本存在
if [ ! -f "eval.py" ]; then
    echo "Error: Evaluation script not found at eval/eval.py"
    echo "Please download the GQA evaluation script from the official repository"
    exit 1
fi

python eval.py --tier testdev_balanced

# 保存结果副本（可选）
# cp testdev_balanced_predictions.json ${ANS_DIR}/

# 删除临时预测文件
rm -f testdev_balanced_predictions.json

echo "Evaluation completed!"