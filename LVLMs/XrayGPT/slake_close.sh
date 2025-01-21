
ROOT_PATH=/data/aofei
dataset=Slake

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers


python eval_batch.py --num-chunks 4  --model-name no_model \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/slake_qa_pairs.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    --answers-file ${ROOT_PATH}/hallucination/${dataset}/inference/XrayGPT/slake_res.jsonl