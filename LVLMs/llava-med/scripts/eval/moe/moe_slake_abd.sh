
ROOT_PATH=/data/aofei
dataset=Slake

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

visual_enhance_ratio=0.15
bbox_ratio=0.1
epoch_num=6
# dir=llava_med/top_heads/abd/expert4_8_rank16/lora_bbox_${bbox_ratio}/epoch${epoch_num}
# dir=llava_med/moe_ratio_bbox/abd/expert4_8_rank16/lora_bbox_${bbox_ratio}/epoch${epoch_num}
dir=llava_med/top_heads_KL_probe01/abd/expert4_8_rank16/lora_bbox_${bbox_ratio}/epoch${epoch_num}
# dir=llava_med/moe_img_dense_all_query/abd/expert4_8_rank16/use_mask/lora_bbox_${bbox_ratio}/epoch${epoch_num}
# dir=llava_med/moe_img_dense_all_query/abd_visual_prompts/expert4_8_rank16/use_mask/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}
# dir=llava_med/moe_img_query/abd/expert4_8_rank16/use_mask/lora_0.15_bbox_0.1/epoch6
# dir=llava_med/moe/abd/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}

python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test_abd.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    --lora_r 16 \
    --lora_alpha 8 \
    --q_expert_num 4 \
    --k_expert_num 8 \
    --dense_moe True \
    --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

python llava/eval/run_eval.py \
    --gt ${ROOT_PATH}/hallucination/${dataset}/data/test_abd.json \
    --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.txt
