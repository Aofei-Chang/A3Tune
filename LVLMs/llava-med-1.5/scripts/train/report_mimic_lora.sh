#!/bin/bash

seeds=(4)
seed=4
ROOT_PATH=/data/aofei
dataset=MIMIC_CXR

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

export CUDA_VISIBLE_DEVICES=1,2,3

for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    dir="llava_med_1.5/lora_report/epoch12/seed${seed}"

    # torchrun --nnodes=1 --nproc_per_node=3 --master_port=25001 llava/train/train_mem.py \
    #     --model_name_or_path ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
    #     --lora_enable True --lora_r 64 --lora_alpha 16 --mm_projector_lr 2e-5 \
    #     --version v1 \
    #     --tune_mm_mlp_adapter False \
    #     --data_path ${ROOT_PATH}/hallucination/${dataset}/data_report/training.json \
    #     --image_folder ${ROOT_PATH}/hallucination/MIMIC_CXR/sampled_files_train \
    #     --vision_tower openai/clip-vit-large-patch14-336 \
    #     --mm_projector_type mlp2x_gelu \
    #     --mm_vision_select_layer -2 \
    #     --mm_use_im_start_end False \
    #     --mm_use_im_patch_token False \
    #     --image_aspect_ratio pad \
    #     --group_by_modality_length True \
    #     --bf16 True \
    #     --output_dir ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints \
    #     --num_train_epochs 12 \
    #     --per_device_train_batch_size 1 \
    #     --per_device_eval_batch_size 4 \
    #     --gradient_accumulation_steps 8 \
    #     --evaluation_strategy "no" \
    #     --save_strategy "steps" \
    #     --save_steps 1000 \
    #     --save_total_limit 3 \
    #     --learning_rate 2e-4 \
    #     --weight_decay 0. \
    #     --warmup_ratio 0.03 \
    #     --lr_scheduler_type "cosine" \
    #     --logging_steps 1 \
    #     --tf32 True \
    #     --model_max_length 1024 \
    #     --gradient_checkpointing False \
    #     --lazy_preprocess True \
    #     --report_to wandb \
    #     --seed $seed
    #     # --fsdp "full_shard auto_wrap" \
    #     # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    # wait

    # python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
    #     --question-file ${ROOT_PATH}/hallucination/${dataset}/data_report/test.json \
    #     --image-folder ${ROOT_PATH}/hallucination/mimic_cxr/images \
    #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    #     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    # dir="llava_med_1.5/lora_report/original"
    # python llava/eval/eval_batch.py --num-chunks 4  ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
    #     --question-file ${ROOT_PATH}/hallucination/${dataset}/data_report/test.json \
    #     --image-folder ${ROOT_PATH}/hallucination/mimic_cxr/images \
    #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_original.jsonl

    
    # conda activate report_eval2
    # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
    #     --gt ${ROOT_PATH}/hallucination/${dataset}/data_report/test.csv \
    #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.csv \
    #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.csv \
    #     --report True
    # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
    #     --model_answers_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    #     --eval_res_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_all_metrics.txt \
    #     --RadGraphFile ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.csv

    # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
    #     --gt ${ROOT_PATH}/hallucination/${dataset}/data_report/test.csv \
    #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_original.csv \
    #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_original.csv \
    #     --report True
    # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
    #     --model_answers_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_original.jsonl \
    #     --eval_res_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_all_metrics_original.txt \
    #     --RadGraphFile ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_original.csv

    

#     baselines=(PAI m3id avisc VCD DoLa) 
    baselines=(opera) 
#     # baselines=(beam greedy nucleus) 
#     # baselines=(m3id avisc)
#     # conda activate report_eval2
    for baseline in "${baselines[@]}"; do
        python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
            --question-file ${ROOT_PATH}/hallucination/${dataset}/data_report/original_inference/test.json \
            --image-folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
            --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
            --baseline ${baseline} \
            --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

        # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
        #     --gt ${ROOT_PATH}/hallucination/${dataset}/data_report/test.csv \
        #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.csv \
        #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_${baseline}.csv \
        #     --report True

        # python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
        #         --model_answers_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        #         --eval_res_file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_all_metrics_${baseline}.txt \
        #         --RadGraphFile ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_${baseline}.csv

        # wait
    done
done