
ROOT_PATH=/data/aofei
dataset=Slake

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

visual_enhance_ratio=0.08
bbox_ratio=0.03
epoch_num=6

# dir=llava_med/moe_img_dense_all_query/all_expert_8_16_rank16/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}
# dir=llava_med/moe_img_dense_all_query/all_expert_8_16_rank16/lora_bbox_${bbox_ratio}/epoch${epoch_num}
# dir=llava_med/moe_img_dense_all_query/all_expert_4_8_rank16/lora_bbox_${bbox_ratio}/epoch${epoch_num}
# dir=llava_med/moe_img_dense_all_query/top32/all_expert_8_16_rank16/lora_bbox_${bbox_ratio}/epoch${epoch_num}
# dir=llava_med/top_heads_KL_probe01/all/expert8_16_rank16/lora_bbox_${bbox_ratio}/epoch${epoch_num}
# dir=llava_med/top_heads_simple/all/expert4_8_rank16/lora_bbox_${bbox_ratio}/epoch${epoch_num}
# dir=llava_med/top_heads_simple_VE/all/expert4_8_rank16/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}
seed=2024
dir=llava_med/ours_v0/all_expert_8_16_rank16/lora_${visual_enhance_ratio}_bbox_${bbox_ratio}/epoch${epoch_num}_seed${seed}
# dir=llava_med/moe_img_dense_all_query/all_expert_8_16_rank16/lora_0.08_bbox_0.03/epoch6


python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
    --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    --lora_r 16 \
    --lora_alpha 8 \
    --q_expert_num 8 \
    --k_expert_num 16 \
    --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

python llava/eval/run_eval.py \
    --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
    --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.txt
