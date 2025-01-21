
ROOT_PATH=/data/aofei
dataset=Slake

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

visual_enhance_ratio=0.2
bbox_ratio=0.4
epoch_num=6

dir=llava_med/moe_query_gt/only_abd02_top1/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}
# dir=llava_med/moe_query/lora_0.15_bbox_0.1/epoch6

python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test_abd.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

python llava/eval/run_eval.py \
    --gt ${ROOT_PATH}/hallucination/${dataset}/data/test_abd.json \
    --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.txt
