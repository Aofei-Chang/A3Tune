
ROOT_PATH=/data/aofei
dataset=VQA_RAD

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

visual_enhance_ratio=0.2
bbox_ratio=0.1
epoch_num=9

# dir=llava_med/visual_enhance/lung/lora_0.1_bbox_0.05/epoch15
dir=llava_med/visual_enhance/lung/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}

python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test_lung.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
    --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

python llava/eval/run_eval.py \
    --gt ${ROOT_PATH}/hallucination/${dataset}/data/test_lung.json \
    --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.txt
