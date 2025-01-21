ROOT_PATH=/data/aofei
dataset=Slake

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers
epoch_num=3
dir=llava_1.5/visual/lung/bbox0.05_visual0.1/epoch${epoch_num}

python llava/eval/eval_batch.py --num-chunks 4  --model-base ${ROOT_PATH}/LLM/llava-v1.5 \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test_lung.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    --model-name ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

python llava/eval/run_eval.py \
    --gt ${ROOT_PATH}/hallucination/${dataset}/data/test_lung.json \
    --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.txt
