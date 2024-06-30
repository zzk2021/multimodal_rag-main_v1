#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"

model=${1:-"liuhaotian/llava-v1.5-7b"}
name=$model
python -m llava.eval.model_vqa_mmbench \
    --model-path /mnt/data/xue.w/yf/checkpoint/$model \
    --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/$model.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama3

mkdir -p playground/data/eval/mmbench_cn/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment $model 
