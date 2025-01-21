#!/bin/bash

seeds=(4 8)
ROOT_PATH=/data/aofei
dataset=Slake

# export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers
export CUDA_VISIBLE_DEVICES=0,1,2,3
epoch_num=9

for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    dir=llava_med/gt_exp/lora

    torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
        llava/train/train_mem.py \
        --model_name_or_path ${ROOT_PATH}/LLM/llava_med \
        --lora_enable True \
        --mm_projector_lr 2e-5 \
        --tune_mm_mlp_adapter False \
        --data_path ${ROOT_PATH}/hallucination/${dataset}/data/with_gt/training_only_weak.json \
        --image_folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
        --vision_tower openai/clip-vit-large-patch14 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end True \
        --bf16 True \
        --output_dir ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints \
        --num_train_epochs $epoch_num \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 3 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 1024 \
        --gradient_checkpointing False \
        --lazy_preprocess True \
        --use_bbox False \
        --visual_focus False \
        --use_moe False \
        --use_mask False \
        --seed $seed \
        # --fsdp "full_shard auto_wrap" \
        # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \

    wait

    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/data/with_gt/test_with_gt.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
        --baseline beam \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/${dataset}/data/with_gt/test_with_gt.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_beam.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_beam.txt

    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/data/with_gt/test_with_gt.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
        --baseline greedy \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/${dataset}/data/with_gt/test_with_gt.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_greedy.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_greedy.txt

    # python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
    #     --question-file ${ROOT_PATH}/hallucination/${dataset}/data/with_gt/test_with_gt.json \
    #     --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    #     --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    # python llava/eval/run_eval.py \
    #     --gt ${ROOT_PATH}/hallucination/${dataset}/data/with_gt/test_with_gt.json \
    #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.txt


    # python llava/eval/eval_batch.py --num-chunks 3  --model-name ${ROOT_PATH}/LLM/llava_med \
    #     --question-file ${ROOT_PATH}/hallucination/${dataset}/data/with_gt/test_with_gt.json \
    #     --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
    #     --baseline greedy \
    #     --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_ori_greedy.jsonl 

    # python llava/eval/run_eval.py \
    #     --gt ${ROOT_PATH}/hallucination/${dataset}/data/with_gt/test_with_gt.json \
    #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_ori_greedy.jsonl \
    #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_ori_greedy.txt


    wait
done


       