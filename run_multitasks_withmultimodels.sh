#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
TASKS=("mmbench_cn" )   # 要跑的任务 vizwiz_vqa_val gqa  seedbench_2_plus  mmbench_cn  mmbench_en pope_adv  pope_pop  pope_random
MODEL_DIR="/datas/wangzijing02/research/multimodal/Vision-Function-Layer-main/merged_models_7B/mmcn/2"

# 自动读取该目录下所有模型文件夹
MODELS=($(ls "${MODEL_DIR}"))

# 如果需要 drop，可以定义数组，否则设置为空数组
DROPS=({4..10})

for TASK in "${TASKS[@]}"
do
    echo ">>> Running TASK ${TASK}..."

    # 遍历每个模型
    for MODEL in "${MODELS[@]}"
    do
        echo ">>>   Running MODEL ${MODEL}..."

        for DROP in "${DROPS[@]}"
        do
            echo ">>>     Running DROP ${DROP}..."

            python3 -m accelerate.commands.launch \
                --main_process_port 17071 \
                --num_processes=8 \
                -m lmms_eval \
                --model qwen2_5_vl \
                --model_args pretrained="${MODEL_DIR}/${MODEL},use_flash_attention_2=True" \
                --tasks ${TASK} \
                --batch_size 1 \
                --drop ${DROP} \
                --log_samples \
                --log_samples_suffix pro-${MODEL}-${TASK}-drop${DROP} \
                --output_path ./logs/${TASK}/${MODEL}/${TASK}-drop${DROP}

            echo ">>>     Finished DROP ${DROP}"
            echo "--------------------------------"
        done

        echo ">>>   Finished MODEL ${MODEL}"
    done

    echo ">>> Finished TASK ${TASK}"
    echo "================================"
done
