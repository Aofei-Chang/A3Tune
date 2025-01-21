ROOT_PATH=/data/aofei
dataset=VQA_RAD

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

# dir=llava_med/lora/visual_prompt/abd/epoch6
# dir=llava_med/original/baselines/opera
# dir=llava_med/lora/epoch6_v2/0.1_data/seed4/
dir=llava_med/lora/epoch9/seed4

python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
    --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/baselines/VCD/pred.jsonl \
    --baseline VCD \
    --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

python llava/eval/run_eval.py \
    --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
    --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/baselines/VCD/pred.jsonl \
    --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/baselines/VCD/eval_res.txt