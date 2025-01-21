seeds=(4)
ROOT_PATH=/data/aofei
# baselines=(DoLa PAI opera beam greedy nucleus)
# baselines=(VCD)
baselines=(original)
# baselines=(avisc m3id)
dataset=Slake
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
        --baseline ${baseline}

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/eval_res_${baseline}.txt

    wait
done

# dataset=VQA_RAD
# for baseline in "${baselines[@]}"; do
#     python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
#         --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#         --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
#         --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
#         --baseline ${baseline}

#     python llava/eval/run_eval.py \
#         --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
#         --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
#         --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/eval_res_${baseline}.txt

#     wait
# done

dataset=PathVQA/pvqa
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/images/test \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
        --baseline ${baseline}

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/${dataset}/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/eval_res_${baseline}.txt

    wait
done

# dataset=IU_Xray
# for baseline in "${baselines[@]}"; do
#     python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
#         --question-file ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
#         --image-folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
#         --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
#         --baseline ${baseline}

#     python llava/eval/run_eval.py \
#         --gt ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
#         --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
#         --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/eval_res_${baseline}.txt

#     wait
# done


# dataset=OmniMedVQA
# for baseline in "${baselines[@]}"; do
#     python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
#         --question-file ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
#         --image-folder ${ROOT_PATH}/hallucination/${dataset}/VQA/raw/OmniMedVQA \
#         --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
#         --baseline ${baseline}

#     python llava/eval/run_eval.py \
#         --gt ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
#         --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
#         --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/eval_res_${baseline}.txt \
#         --mc True
#     wait
# done


# dataset=OL3I
# for baseline in "${baselines[@]}"; do
#     python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava_med \
#         --question-file ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
#         --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
#         --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
#         --baseline ${baseline}

#     python llava/eval/run_eval.py \
#         --gt ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
#         --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/pred_${baseline}.jsonl \
#         --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/ori_inference/eval_res_${baseline}.txt \
#         --mc True

#     wait
# done