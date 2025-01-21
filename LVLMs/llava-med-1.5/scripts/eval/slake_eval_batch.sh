
ROOT_PATH=/data/aofei
dataset=Slake

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers


python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/slake_qa_pairs.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    --answers-file ${ROOT_PATH}/hallucination/${dataset}/inference/llava_med_1.5/slake_res.jsonl