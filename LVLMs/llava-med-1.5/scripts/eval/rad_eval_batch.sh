
ROOT_PATH=/data/aofei
dataset=VQA_RAD

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

dir=lora/epoch3

python llava/eval/eval_batch.py --num-chunks 1  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
    --peft-path ${ROOT_PATH}//hallucination/mitigation/${dataset}/${dir}/checkpoints \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
    --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl

python llava/eval/run_eval.py \
    --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
    --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.txt
