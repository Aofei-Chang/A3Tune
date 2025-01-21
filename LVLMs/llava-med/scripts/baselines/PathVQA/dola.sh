ROOT_PATH=/data/aofei
dataset=PathVQA/pvqa

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

seed=4
dir="llava_med/lora/epoch3/seed${seed}"

python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/test.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/images/test \
    --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/baselines/DoLa/pred.jsonl \
    --baseline DoLa \
    --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

python llava/eval/run_eval.py \
    --gt ${ROOT_PATH}/hallucination/${dataset}/test.json \
    --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/baselines/DoLa/pred.jsonl \
    --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/baselines/DoLa/eval_res.txt