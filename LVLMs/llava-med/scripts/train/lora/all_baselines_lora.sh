seeds=(4)
ROOT_PATH=/data/aofei
# baselines=(DoLa PAI opera beam greedy nucleus)
# baselines=(avisc m3id)
baselines=(m3id)
seed=4
dataset=Slake
dir="llava_med/lora/epoch6_v2/seed${seed}"
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 3  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/imgs \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --baseline ${baseline} \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_${baseline}.txt
    wait
done

dataset=VQA_RAD
dir="llava_med/lora/epoch9/seed${seed}"
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 3  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --baseline ${baseline} \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    # python llava/eval/run_eval.py \
    #     --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
    #     --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred.jsonl \
    #     --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res.txt
    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_${baseline}.txt

    wait
done

dataset=PathVQA/pvqa
dir="llava_med/lora/epoch3/seed${seed}"
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 3  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/${dataset}/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/images/test \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --baseline ${baseline} \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/${dataset}/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_${baseline}.txt

    wait
done

dataset=IU_Xray
dir="llava_med/lora/epoch6/seed${seed}"
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 3  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --baseline ${baseline} \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_${baseline}.txt

    wait
done


dataset=OmniMedVQA
dir="llava_med/lora/epoch3/seed${seed}"
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 3  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/VQA/raw/OmniMedVQA \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --baseline ${baseline} \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_${baseline}.txt \
        --mc True
    wait
done


dataset=OL3I
dir="llava_med/lora/epoch4/seed${seed}"
for baseline in "${baselines[@]}"; do
    python llava/eval/eval_batch.py --num-chunks 3  --model-name ${ROOT_PATH}/LLM/llava_med \
        --question-file ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
        --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
        --answers-file ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --baseline ${baseline} \
        --peft-path ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/checkpoints

    python llava/eval/run_eval.py \
        --gt ${ROOT_PATH}/hallucination/CARES/${dataset}/test.json \
        --pred ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
        --eval_res ${ROOT_PATH}/hallucination/mitigation/${dataset}/${dir}/inference/eval_res_${baseline}.txt \
        --mc True

    wait
done


